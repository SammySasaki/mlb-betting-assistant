from sqlalchemy.orm import Session
from typing import Optional
from app.db_models import Prediction, Game
from app.interfaces.ipredictions_repository import IPredictionRepository
from datetime import date
from typing import List
from sqlalchemy import or_


class SqlAlchemyPredictionRepository(IPredictionRepository):
    def __init__(self, session: Session):
        self.session = session

    def get_by_game_and_market(self, game_id: int, market: str) -> Optional[Prediction]:
        return (
            self.session.query(Prediction)
            .filter_by(game_id=game_id, market=market)
            .first()
        )

    def upsert_total_prediction(self, game_id: int, predicted_total: float, edge: float, recommendation: str) -> Prediction:
        prediction = self.get_by_game_and_market(game_id, "TOTAL")
        if prediction:
            prediction.predicted_total_runs = predicted_total
            prediction.edge = edge
            prediction.recommendation = recommendation
        else:
            prediction = Prediction(
                game_id=game_id,
                predicted_total_runs=predicted_total,
                edge=edge,
                recommendation=recommendation,
                market="TOTAL",
            )
            self.session.add(prediction)
        return prediction

    def upsert_ml_prediction(self, game_id: int, home_win_prob: float, away_win_prob: float, recommendation: str) -> Prediction:
        prediction = self.get_by_game_and_market(game_id, "H2H")
        if prediction:
            prediction.home_win_prob = home_win_prob
            prediction.away_win_prob = away_win_prob
            prediction.recommendation = recommendation
        else:
            prediction = Prediction(
                game_id=game_id,
                home_win_prob=home_win_prob,
                away_win_prob=away_win_prob,
                recommendation=recommendation,
                market="H2H",
            )
            self.session.add(prediction)
        return prediction

    def commit(self) -> None:
        self.session.commit()

    def get_predictions(
        self,
        game_date: date,
        team: Optional[str] = None,
        market: Optional[str] = None,
        exclude_no_bet: bool = False,
    ) -> List[Prediction]:
        query = self.session.query(Prediction).join(Game).filter(Game.date == game_date)

        if team and team != "ANY":
            query = query.filter(or_(Game.home_team == team, Game.away_team == team))

        if market and market != "ANY":
            query = query.filter(Prediction.market == market)

        if exclude_no_bet:
            query = query.filter(Prediction.recommendation != "NO BET")

        return query.all()