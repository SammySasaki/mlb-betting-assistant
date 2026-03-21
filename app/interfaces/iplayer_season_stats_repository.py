from abc import ABC, abstractmethod
from typing import List, Optional

from app.db.models import PlayerSeasonStats


class IPlayerSeasonStatsRepository(ABC):

    @abstractmethod
    def get_by_player_and_season(self, player_id: int, season_year: int) -> Optional[PlayerSeasonStats]:
        pass

    @abstractmethod
    def get_by_season(self, season_year: int) -> List[PlayerSeasonStats]:
        pass

    @abstractmethod
    def get_by_team_and_season(self, team: str, season_year: int) -> List[PlayerSeasonStats]:
        pass

    @abstractmethod
    def upsert(self, stats: PlayerSeasonStats) -> None:
        pass
