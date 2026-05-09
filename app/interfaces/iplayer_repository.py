from abc import ABC, abstractmethod
from typing import Optional
from app.db.models import Player

class IPlayerRepository(ABC):
    """Repository interface for player name lookup"""

    @abstractmethod
    def get_player_name(self, player_id: int) -> Optional[str]:
        """
        Given a player_id, return the player's name.
        """
        pass

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[Player]:
        """Return the Player whose name matches (case-insensitive), or None."""
        pass

    @abstractmethod
    def get_by_id(self, player_id: int) -> Optional[Player]:
        """
        Given a player_id, return the Player object or None if not found.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Commit all pending player upserts."""
        pass

    @abstractmethod
    def upsert(self, player_id: int, name: str, team: str, position: str, throwing_hand: Optional[str] = None, existing: Optional[Player] = None) -> None:
        """
        Insert or update a player record. For new players, throwing_hand is set if provided.
        For existing players, only team and position are updated.
        """
        pass