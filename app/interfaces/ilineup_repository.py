from abc import ABC, abstractmethod
from typing import List, Optional
from datetime import date

from app.db_models import Lineups


class ILineupRepository(ABC):
    """Repository interface for LineupEntry persistence"""

    @abstractmethod
    def get_lineup_by_game(self, game_id: int) -> List[Lineups]:
        pass

    @abstractmethod
    def upsert_lineup_entries(self, game_id: int, entries: List[Lineups]) -> None:
        """Insert or update lineup entries for a game"""
        pass

    @abstractmethod
    def delete_lineup_for_game(self, game_id: int) -> None:
        pass

    @abstractmethod
    def get_lineups_by_date(self, game_date: date) -> List[Lineups]:
        pass

    @abstractmethod
    def get_lineups_by_date_by_team(self, game_date: date, team: str) -> List[Lineups]:
        """Get all lineup entries for a specific team on a given date"""
        pass