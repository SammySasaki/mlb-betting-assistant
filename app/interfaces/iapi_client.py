from abc import ABC, abstractmethod
from typing import Dict, Any

class IApiClient(ABC):
    """Interface for fetching lineups from an external API"""

    @abstractmethod
    def get_boxscore(self, game_id: int) -> Dict[str, Any]:
        """Fetch boxscore for a single game"""
        pass