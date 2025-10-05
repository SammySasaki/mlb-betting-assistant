import pandas as pd
from datetime import date, timedelta
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.sql import func
import joblib
from app.db_models import Game, Prediction
from app.services.feature_extractor import FeatureExtractor
from app.interfaces.ipredictions_repository import IPredictionRepository
from app.interfaces.igame_repository import IGameRepository


class PredictionService:
    def __init__(
        self,
        extractor: FeatureExtractor,
        prediction_repo: IPredictionRepository,
        game_repo: IGameRepository,
        totals_model_path: str = "app/models/mlb_ridge.pkl",
        ml_model_path: str = "app/models/moneyline_logreg.pkl",
    ):
        self.extractor = extractor
        self.prediction_repo = prediction_repo
        self.game_repo = game_repo

        self.totals_model = joblib.load(totals_model_path)
        self.ml_model = joblib.load(ml_model_path)

        # Feature sets (match training pipeline)
        self.numeric_feature_cols = [
            "temperature",
            "wind_flag",
            "home_avg_runs_vs_arm", "home_avg_runs_lastx_total",
            "away_avg_runs_vs_arm", "away_avg_runs_lastx_total",
            "home_sp_era", "away_sp_era",
            "home_sp_whip", "away_sp_whip",
            "home_bullpen_era", "away_bullpen_era",
            "home_lineup_ops", "away_lineup_ops",
            "venue_run_factor",
            "home_sp_vs_away_lineup",
            "away_sp_vs_home_lineup",
            "home_offense_vs_away_bullpen",
            "away_offense_vs_home_bullpen",
        ]

        self.ml_categorical = [
            "home_throwing_hand", "away_throwing_hand"
        ]
        self.ml_numerical = [
            "home_sp_era", "away_sp_era",
            "home_sp_whip", "away_sp_whip",
            "home_sp_last3_era", "away_sp_last3_era",
            "home_bullpen_era", "away_bullpen_era",
            "home_lineup_ops", "away_lineup_ops",
            "home_avg_runs_lastx_total", "away_avg_runs_lastx_total",
            "home_avg_runs_vs_arm", "away_avg_runs_vs_arm",
            "home_team_wins", "home_team_losses",
            "away_team_wins", "away_team_losses",
            "home_run_diff", "away_run_diff",
            "home_team_wins_last10", "away_team_wins_last10",
        ]

    # ----------------------------
    # Totals prediction
    # ----------------------------
    def predict_total_runs(self, features: dict) -> float:
        numeric_features = {k: v for k, v in features.items()
                            if k in self.numeric_feature_cols}
        X = pd.DataFrame([numeric_features], columns=self.numeric_feature_cols)
        prediction = self.totals_model.predict(X)[0]
        return round(float(prediction), 2)

    def recommend_bet(self, features: dict, total_line: float, threshold: float = 0.5) -> dict:
        predicted_total = self.predict_total_runs(features)
        diff = predicted_total - total_line

        if diff > threshold:
            recommendation = "BET OVER"
        elif diff < -threshold:
            recommendation = "BET UNDER"
        else:
            recommendation = "NO BET"

        return {
            "predicted_total": predicted_total,
            "market_line": total_line,
            "edge": round(diff, 2),
            "recommendation": recommendation
        }

    # ----------------------------
    # Moneyline prediction
    # ----------------------------
    def predict_home_win_prob(self, ml_features: dict) -> float | None:
        required_features = self.ml_categorical + self.ml_numerical
        for f in required_features:
            if f not in ml_features or ml_features[f] is None:
                print("missing feature: " + f)
                return None

        new_game = pd.DataFrame([ml_features])
        home_win_prob = self.ml_model.predict_proba(new_game)[0, 1]
        return float(home_win_prob)

    def recommend_ml_bet(self, ml_features: dict, home_dec: float, away_dec: float, bankroll_units: int = 100) -> dict | None:
        home_win_prob = self.predict_home_win_prob(ml_features)
        if home_win_prob is None:
            return None

        away_win_prob = 1 - home_win_prob

        def kelly_fraction(p: float, odds: float) -> float:
            b = odds - 1
            q = 1 - p
            f = (b * p - q) / b
            return max(f, 0)

        home_kelly = kelly_fraction(home_win_prob, home_dec)
        away_kelly = kelly_fraction(away_win_prob, away_dec)

        home_units = home_kelly * bankroll_units
        away_units = away_kelly * bankroll_units

        if home_units > 0 and home_units > away_units:
            recommendation = f"BET {home_units:.2f}U HOME"
        elif away_units > 0 and away_units > home_units:
            recommendation = f"BET {away_units:.2f}U AWAY"
        else:
            recommendation = "NO BET"

        return {
            "home_win_prob": home_win_prob,
            "away_win_prob": away_win_prob,
            "recommendation": recommendation,
        }

    # ----------------------------
    # Game fetchers (delegate to GameRepository ideally)
    # ----------------------------
    def fetch_upcoming_games(self):
        return self.game_repo.get_upcoming_games()

    def fetch_old_games(self, n: int):
        return self.game_repo.get_old_games(n)

    # ----------------------------
    # Run predictions & log
    # ----------------------------
    def run(self):
        games = self.fetch_upcoming_games()
        results, ml_results = [], []

        for game in games:
            features = self.extractor.extract_total_features_from_game(game)
            ml_features = self.extractor.extract_ML_features_from_game(game)

            if features is None:
                print(f"Skipping game {game.id} due to missing total features")
                continue
            if ml_features is None:
                print(f"Skipping game {game.id} due to missing ML features")
                continue

            if game.hr_total_runs_line is not None:
                result = self.recommend_bet(features, game.hr_total_runs_line)
                result["game_id"] = game.id
                results.append(result)

            if game.home_ml_price is not None and game.away_ml_price is not None:
                ml_result = self.recommend_ml_bet(ml_features, game.home_ml_price, game.away_ml_price)
                if ml_result:
                    ml_result["game_id"] = game.id
                    ml_results.append(ml_result)

        # Save predictions via repository
        for result in results:
            print('inserting total prediction for game', result["game_id"])
            self.prediction_repo.upsert_total_prediction(
                game_id=result["game_id"],
                predicted_total=result["predicted_total"],
                edge=result["edge"],
                recommendation=result["recommendation"],
            )
        for ml_result in ml_results:
            self.prediction_repo.upsert_ml_prediction(
                game_id=ml_result["game_id"],
                home_win_prob=ml_result["home_win_prob"],
                away_win_prob=ml_result["away_win_prob"],
                recommendation=ml_result["recommendation"],
            )

        self.prediction_repo.commit()

    # ----------------------------
    # Predict + log for a single game
    # ----------------------------
    def predict_and_log_for_game(self, game):
        """
        Extract features, run predictions, and log results for a single game.
        Returns a dict with 'totals' and 'ml' results.
        """
        results, ml_results = None, None

        # --- Extract features ---
        features = self.extractor.extract_total_features_from_game(game)
        ml_features = self.extractor.extract_ML_features_from_game(game)

        if features is None:
            print(f"Skipping totals prediction for game {game.id} (missing features)")
        else:
            if game.hr_total_runs_line is not None:
                results = self.recommend_bet(features, game.hr_total_runs_line)
                results["game_id"] = game.id

                print('inserting total prediction for game', results["game_id"])
                self.prediction_repo.upsert_total_prediction(
                    game_id=game.id,
                    predicted_total=results["predicted_total"],
                    edge=results["edge"],
                    recommendation=results["recommendation"],
                )

        if ml_features is None:
            print(f"Skipping ML prediction for game {game.id} (missing features)")
        else:
            if game.home_ml_price is not None and game.away_ml_price is not None:
                ml_results = self.recommend_ml_bet(
                    ml_features, game.home_ml_price, game.away_ml_price
                )
                if ml_results:
                    ml_results["game_id"] = game.id
                    self.prediction_repo.upsert_ml_prediction(
                        game_id=game.id,
                        home_win_prob=ml_results["home_win_prob"],
                        away_win_prob=ml_results["away_win_prob"],
                        recommendation=ml_results["recommendation"],
                    )

        # Commit once for both
        self.prediction_repo.commit()

        return {"totals": results, "ml": ml_results}
