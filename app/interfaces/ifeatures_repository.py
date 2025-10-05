from abc import ABC, abstractmethod
from typing import Optional
from app.db_models import PitcherFeatures, BullpenFeatures, TeamFeatures

class IFeaturesRepository(ABC):
    @abstractmethod
    def get_by_game_and_player(self, game_id: int, player_id: int) -> Optional[PitcherFeatures]:
        """Fetch pitcher features for a given game and player (probable SP)."""
        pass

    @abstractmethod
    def get_team_features_by_game_and_team(self, game_id: int, team: str) -> Optional[TeamFeatures]:
        """Fetch team features for a given game and team."""
        pass

    @abstractmethod
    def get_bullpen_features_by_game_and_team(self, game_id: int, team: str) -> Optional[BullpenFeatures]:
        """Fetch pitcher features for a given game and player (probable SP)."""
        pass