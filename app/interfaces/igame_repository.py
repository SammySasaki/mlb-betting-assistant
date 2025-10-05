from abc import ABC, abstractmethod
from datetime import date
from typing import List, Optional
from app.db_models import Game


class IGameRepository(ABC):
    """Repository interface for interacting with Game entities."""

    @abstractmethod
    def get_by_id(self, game_id: int) -> Optional[Game]:
        pass

    @abstractmethod
    def get_games_by_date(self, game_date: date) -> List[Game]:
        pass

    @abstractmethod
    def get_games_by_season(self, season_year: int) -> List[Game]:
        pass

    @abstractmethod
    def add(self, game: Game) -> Game:
        pass

    @abstractmethod
    def update(self, game: Game) -> Game:
        pass

    @abstractmethod
    def delete(self, game_id: int) -> None:
        pass

    @abstractmethod
    def get_game_by_date_by_team(self, game_date: date, team: str) -> Optional[Game]:
        """
        Get a single game for a given date and team (either home or away).
        """
        pass

    @abstractmethod
    def get_games_by_team_before_date(
        self, team: str, before_date: date, season_year: int
    ) -> List[Game]:
        """Return all games for a team before a given date in a given season."""
        pass

    @abstractmethod
    def get_team_record_before_date(self, team: str, date: date, season_year: int):
        """Return the win-loss record for a team before a given game."""
        pass