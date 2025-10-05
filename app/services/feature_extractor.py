import numpy as np
from sqlalchemy import func, case
from app.utils.utils import venue_orientations, venue_run_factors
from typing import List, Optional
from app.db_models import Game, Player, TeamFeatures, BullpenFeatures, PitcherFeatures
from app.interfaces.iplayer_game_stats_repository import IPlayerGameStatsRepository
from app.interfaces.igame_repository import IGameRepository
from app.interfaces.ifeatures_repository import IFeaturesRepository
from app.interfaces.iplayer_repository import IPlayerRepository
from app.interfaces.iweather_repository import IWeatherRepository


class FeatureExtractor:
    NEW_PITCHER_ERA = 4.80
    NEW_PITCHER_WHIP = 1.40

    def __init__(self,
                 pgs_repo: IPlayerGameStatsRepository, 
                 game_repo: IGameRepository, 
                 feature_repo: IFeaturesRepository,
                 player_repo: IPlayerRepository,
                 weather_repo: IWeatherRepository):
        self.pgs_repo = pgs_repo
        self.game_repo = game_repo
        self.feature_repo = feature_repo
        self.player_repo = player_repo
        self.weather_repo = weather_repo

    def extract_ML_features_from_game(self, game: Game) -> dict:
        """Extracts precomputed pregame features for a single MLB game for Moneyline predictions."""

        features = {
            "game_id": game.id,
            "date": game.date,
            "start_hour_utc": game.start_hour_utc,
            "season_year": game.season_year,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "venue": game.venue,
        }

        # Home team season record before game
        home_record = self.game_repo.get_team_record_before_date(game.home_team, game.date, game.season_year)

        features["home_team_wins"] = home_record[0] or 0
        features["home_team_losses"] = home_record[1] or 0

        

        # Away team season record before game
        away_record = self.game_repo.get_team_record_before_date(game.away_team, game.date, game.season_year)

        features["away_team_wins"] = away_record[0] or 0
        features["away_team_losses"] = away_record[1] or 0


        # --- Team features ---
        home_team_feats = self.feature_repo.get_team_features_by_game_and_team(game.id, game.home_team)
        away_team_feats = self.feature_repo.get_team_features_by_game_and_team(game.id, game.away_team)

        if home_team_feats:
            features.update({
                "home_team_wins_last10": home_team_feats.wins_last10,
                "home_avg_runs_lastx_total": home_team_feats.avg_runs_last10,
                "home_run_diff": home_team_feats.run_diff_season
            })

        if away_team_feats:
            features.update({
                "away_team_wins_last10": away_team_feats.wins_last10,
                "away_avg_runs_lastx_total": away_team_feats.avg_runs_last10,
                "away_run_diff": away_team_feats.run_diff_season
            })
        # --- Bullpen features ---
        home_bp = self.feature_repo.get_bullpen_features_by_game_and_team(game.id, game.home_team)
        away_bp = self.feature_repo.get_bullpen_features_by_game_and_team(game.id, game.away_team)

        if home_bp:
            features["home_bullpen_era"] = home_bp.bullpen_era
        if away_bp:
            features["away_bullpen_era"] = away_bp.bullpen_era

        # --- Pitcher features ---
        home_sp = self.feature_repo.get_by_game_and_player(game.id, game.home_probable_pitcher_id)
        away_sp = self.feature_repo.get_by_game_and_player(game.id, game.away_probable_pitcher_id)

        if home_sp:
            features.update({
                "home_sp_era": home_sp.sp_era if home_sp.sp_era is not None else self.NEW_PITCHER_ERA,
                "home_sp_whip": home_sp.sp_whip if home_sp.sp_whip is not None else self.NEW_PITCHER_WHIP,
                "home_sp_last3_era": home_sp.last3_era if home_sp.last3_era is not None else self.NEW_PITCHER_ERA,
            })

        if away_sp:
            features.update({
                "away_sp_era": away_sp.sp_era if away_sp.sp_era is not None else self.NEW_PITCHER_ERA,
                "away_sp_whip": away_sp.sp_whip if away_sp.sp_whip is not None else self.NEW_PITCHER_WHIP,
                "away_sp_last3_era": away_sp.last3_era if away_sp.last3_era is not None else self.NEW_PITCHER_ERA,
            })

        home_pitcher = self.player_repo.get_by_id(game.home_probable_pitcher_id)
        if home_pitcher:
            features["home_throwing_hand"] = home_pitcher.throwing_hand

        away_pitcher = self.player_repo.get_by_id(game.away_probable_pitcher_id)
        if away_pitcher:
            features["away_throwing_hand"] = away_pitcher.throwing_hand

        
        features["home_lineup_ops"] = self._lineup_ops(game.home_team, game.date, game.season_year)
        features["away_lineup_ops"] = self._lineup_ops(game.away_team, game.date, game.season_year)

        # --- Runs vs arm ---
        features["home_avg_runs_vs_arm"] = self._runs_vs_arm(game.home_team, game.away_probable_pitcher, game.date, game.season_year, home=True)
        features["away_avg_runs_vs_arm"] = self._runs_vs_arm(game.away_team, game.home_probable_pitcher, game.date, game.season_year, home=False)

        # --- Derived / interaction features ---
        if home_sp and away_sp:
            home_era = home_sp.sp_era if home_sp.sp_era is not None else self.NEW_PITCHER_ERA
            away_era = away_sp.sp_era if away_sp.sp_era is not None else self.NEW_PITCHER_ERA
            features["sp_era_diff"] = home_era - away_era
        else:
            features["sp_era_diff"] = 0.0
        features["run_diff_diff"] = (home_team_feats.run_diff_season or 0) - (away_team_feats.run_diff_season or 0)
        features["lineup_ops_diff"] = features["home_lineup_ops"] - features["away_lineup_ops"]


        return features

    def extract_total_features_from_game(self, game: Game) -> dict:
        features = {
            "game_id": game.id,
            "date": game.date,
            "start_hour_utc": game.start_hour_utc,
            "season_year": game.season_year,
            "home_team": game.home_team,
            "away_team": game.away_team,
            "venue": game.venue,
        }
        
        features.update(self._extract_weather(game))

        field_orientation = venue_orientations.get(features["venue"], 0) 
        wind_dir = features.get("wind_direction")
        features["wind_direction"] = float(wind_dir) if wind_dir is not None else 0.0
        
        wind_rel_angle = (features["wind_direction"] - field_orientation) % 360
        def wind_flag(angle):
            if (angle >= 315 or angle <= 45):
                return 1 
            elif 135 <= angle <= 225:
                return -1 
            else:
                return 0
        features["wind_flag"] = wind_flag(wind_rel_angle)
        
        features["home_avg_runs_vs_arm"] = self._runs_vs_arm(game.home_team, game.away_probable_pitcher, game.date, game.season_year, home=True)
        features["away_avg_runs_vs_arm"] = self._runs_vs_arm(game.away_team, game.home_probable_pitcher, game.date, game.season_year, home=False)
        features["home_avg_runs_last10_total"] = self._avg_runs_last_n(game.home_team, game.date, game.season_year, n=10, is_home=True)
        features["away_avg_runs_last10_total"] = self._avg_runs_last_n(game.away_team, game.date, game.season_year, n=10, is_home=False)
        features.update(self._pitcher_stats(game.home_probable_pitcher, game.season_year, prefix="home"))
        features.update(self._pitcher_stats(game.away_probable_pitcher, game.season_year, prefix="away"))

        home_bp = self.feature_repo.get_bullpen_features_by_game_and_team(game.id, game.home_team)
        away_bp = self.feature_repo.get_bullpen_features_by_game_and_team(game.id, game.away_team)
        if home_bp:
            features["home_bullpen_era"] = home_bp.bullpen_era
        if away_bp:
            features["away_bullpen_era"] = away_bp.bullpen_era
        features["home_lineup_ops"] = self._lineup_ops(game.home_team, game.date, game.season_year)
        features["away_lineup_ops"] = self._lineup_ops(game.away_team, game.date, game.season_year)
        features["venue_run_factor"] = venue_run_factors.get(game.venue, 1.0)
        features["home_sp_vs_away_lineup"] = features["home_sp_era"] - features["away_lineup_ops"]
        features["away_sp_vs_home_lineup"] = features["away_sp_era"] - features["home_lineup_ops"]
        features["home_offense_vs_away_bullpen"] = features["home_avg_runs_last10_total"] - features["away_bullpen_era"]
        features["away_offense_vs_home_bullpen"] = features["away_avg_runs_last10_total"] - features["home_bullpen_era"]

        return features


    # ---- Private helpers ----
    def _extract_weather(self, game: Game) -> dict:
        weather = self.weather_repo.get_by_game_id(game.id)
    
        if weather:
            return {
                "temperature": weather.temperature if weather.temperature is not None else np.nan,
                "wind_speed": weather.wind_speed if weather.wind_speed is not None else np.nan,
                "wind_direction": weather.wind_direction if weather.wind_direction is not None else 0.0
            }
        
        # Default values if no weather data
        return {
            "temperature": np.nan,
            "wind_speed": np.nan,
            "wind_direction": 0.0
        }
    
    def _avg_runs_last_n(self, team: str, date, season_year: int, n=10, is_home=True) -> float:
        games = self.game_repo.get_games_by_team_before_date(team, date, season_year)
        games = [g for g in games if (g.home_team == team if is_home else g.away_team == team)]
        games = sorted(games, key=lambda g: g.date, reverse=True)[:n]
        runs = []
        for g in games:
            runs_scored = g.home_score if is_home and g.home_team == team else g.away_score
            runs_scored = g.away_score if not is_home and g.away_team == team else runs_scored
            if runs_scored is not None:
                runs.append(runs_scored)
        return np.mean(runs) if runs else np.nan
    
    def _pitcher_stats(self, pitcher: Player, season_year: int, prefix: str) -> dict:
        stats = self.pgs_repo.get_aggregate_stats(pitcher.id, season_year=season_year)
        if not stats:
            return {
                "era": self.NEW_PITCHER_ERA,
                "whip": self.NEW_PITCHER_WHIP
            }
        total_outs = sum(s.innings_pitched * 3 for s in stats if s.innings_pitched is not None)
        total_er = sum(s.earned_runs for s in stats if s.earned_runs is not None)
        total_hits = sum(s.hits for s in stats if s.hits is not None)
        total_walks = sum(s.walks for s in stats if s.walks is not None)
        
        
        ip = total_outs / 3 if total_outs > 0 else 0
        era = (total_er * 9 / ip) if ip > 0 else None
        whip = ((total_walks + total_hits) / ip) if ip > 0 else None
        return {f"{prefix}_sp_era": era, f"{prefix}_sp_whip": whip}
        
    def _lineup_ops(self, team: str, date, season_year: int) -> float:
        """Calculate OPS for the team's top 5 players lineup over the season up to the given date."""
        stats = self.pgs_repo.get_batters_by_team_and_season_and_date(team, season_year, date)
        player_ops = {}
        for s in stats:
            if s.player_id not in player_ops:
                player_ops[s.player_id] = {"hits":0, "walks":0, "at_bats":0, "doubles":0, "triples":0, "home_runs":0}
            p = player_ops[s.player_id]
            p["hits"] += s.hits or 0
            p["walks"] += s.walks_batting or 0
            p["at_bats"] += s.at_bats or 0
            p["doubles"] += s.doubles or 0
            p["triples"] += s.triples or 0
            p["home_runs"] += s.home_runs or 0
        
        ops_list = []
        for p in player_ops.values():
            if p["at_bats"] >= 50:
                avg = ((p["hits"] + p["walks"]) / max(p["at_bats"] + p["walks"],1)) + \
                    ((p["hits"] - p["doubles"] - p["triples"] - p["home_runs"]) + 2*p["doubles"] + 3*p["triples"] + 4*p["home_runs"]) / max(p["at_bats"],1)
                ops_list.append(avg)
        ops_list.sort(reverse=True)
        top5 = ops_list[:5]
        return np.mean(top5) if top5 else np.nan
    
    def _runs_vs_arm(self, team: str, pitcher: Player, date, season_year: int, home=True) -> float:
        if not pitcher:
            return np.nan
        games = self.game_repo.get_games_by_team_before_date(team, date, season_year)
        runs = []
        for g in games:
            # Only count games vs same-handed pitcher
            ph = pitcher.throwing_hand
            opponent_pitcher = g.away_probable_pitcher if home else g.home_probable_pitcher
            if opponent_pitcher and opponent_pitcher.throwing_hand == ph:
                runs_scored = g.home_score if home and g.home_team == team else g.away_score
                runs_scored = g.away_score if not home and g.away_team == team else runs_scored
                if runs_scored is not None:
                    runs.append(runs_scored)
        return np.mean(runs) if runs else np.nan
    

    if __name__ == "__main__":
        from infra.db.init_db import engine
        from sqlalchemy.orm import Session
        from infra.db.init_db import init_db
        from infra.db.init_db import SessionLocal
        from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
        from app.implementations.sqlalchemy_game_repository import GameRepository
        from app.implementations.sqlalchemy_features_repository import SqlAlchemyFeaturesRepository
        from app.implementations.sqlalchemy_player_repository import SQLAlchemyPlayerRepository
        from app.implementations.sqlalchemy_weather_repository import WeatherRepository
        from app.services.feature_extractor import FeatureExtractor

        with SessionLocal() as session:
            pgs_repo = SqlAlchemyPlayerGameStatsRepository(session)
            game_repo = GameRepository(session)
            feature_repo = SqlAlchemyFeaturesRepository(session)
            player_repo = SQLAlchemyPlayerRepository(session)
            weather_repo = WeatherRepository(session)

            extractor = FeatureExtractor(pgs_repo, game_repo, feature_repo, player_repo, weather_repo)

            game = game_repo.get_by_id(776220)  # Example game ID
            if game:
                ml_features = extractor.extract_ML_features_from_game(game)
                total_features = extractor.extract_total_features_from_game(game)
                print("ML Features:", ml_features)
                print("Total Features:", total_features)