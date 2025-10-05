from abc import ABC, abstractmethod
from typing import Optional
from app.db_models import Player

class IPlayerRepository(ABC):
    """Repository interface for player name lookup"""

    @abstractmethod
    def get_player_name(self, player_id: int) -> Optional[str]:
        """
        Given a player_id, return the player's name.
        """
        pass

    @abstractmethod
    def get_by_id(self, player_id: int) -> Optional[Player]:
        """
        Given a player_id, return the Player object or None if not found.
        """
        pass