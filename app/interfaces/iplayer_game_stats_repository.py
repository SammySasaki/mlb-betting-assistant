from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
from datetime import date

from app.db_models import PlayerGameStats


class IPlayerGameStatsRepository(ABC):
    """Abstract repository interface for PlayerGameStats."""

    @abstractmethod
    def get_by_player_and_season(self, player_id: int, season_year: int) -> List[PlayerGameStats]:
        pass

    @abstractmethod
    def get_aggregate_stats(self, player_id: int, season_year: int) -> Optional[dict]:
        """Return season totals like hits, at_bats, doubles, triples, HR, walks, RBIs."""
        pass

    @abstractmethod
    def get_batters_by_team_and_season_and_date(self, team: str, season_year: int, before_date: date) -> List[PlayerGameStats]:
        pass

    @abstractmethod
    def get_last_n_games_stats(self, player_id: int, n: int, before_date: Optional[date] = None) -> List[PlayerGameStats]:
        pass