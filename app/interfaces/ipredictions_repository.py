from abc import ABC, abstractmethod
from typing import Optional, List
from app.db_models import Prediction
from datetime import date


class IPredictionRepository(ABC):
    """Abstract repository for Prediction model."""

    @abstractmethod
    def get_by_game_and_market(self, game_id: int, market: str) -> Optional[Prediction]:
        pass

    @abstractmethod
    def upsert_total_prediction(self, game_id: int, predicted_total: float, edge: float, recommendation: str) -> Prediction:
        pass

    @abstractmethod
    def upsert_ml_prediction(
        self, game_id: int, home_win_prob: float, away_win_prob: float, recommendation: str
    ) -> Prediction:
        pass

    @abstractmethod
    def commit(self) -> None:
        pass

    @abstractmethod
    def get_predictions(
        self,
        game_date: date,
        team: Optional[str] = None,
        market: Optional[str] = None,
        exclude_no_bet: bool = False,
    ) -> List[Prediction]:
        pass