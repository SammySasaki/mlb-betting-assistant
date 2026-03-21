from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Any

class IApiClient(ABC):
    """Interface for the MLB Stats API."""

    @abstractmethod
    def get_schedule(self, target_date: date) -> list:
        """Return list of game dicts for a given date."""
        pass

    @abstractmethod
    def get_boxscore(self, game_id: int) -> Dict[str, Any]:
        """Fetch boxscore for a single game."""
        pass

    @abstractmethod
    def get_player(self, player_id: int) -> dict | None:
        """Return the people[0] dict for a player, or None if not found."""
        pass

    @abstractmethod
    def get_player_arm(self, player_id: int) -> str | None:
        """Return pitching-hand code ('L'/'R'/'S') for a player, or None."""
        pass

    @abstractmethod
    def get_probable_pitchers(self, game_id: int) -> tuple[int | None, int | None]:
        """Return (home_pitcher_id, away_pitcher_id) for a game."""
        pass