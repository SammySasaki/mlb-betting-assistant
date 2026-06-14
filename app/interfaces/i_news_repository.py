from abc import ABC, abstractmethod
from typing import Optional


class INewsRepository(ABC):

    @abstractmethod
    def upsert_many(self, articles: list[dict]) -> tuple[int, int]:
        """Insert articles, skipping duplicates by URL.

        Returns (inserted_count, skipped_count).
        """
        pass

    @abstractmethod
    def get_recent(self, limit: int = 20, category: Optional[str] = None) -> list:
        """Return recent articles ordered by published_at desc."""
        pass

    @abstractmethod
    def get_recent_by_teams(self, teams: list[str], hours: int = 24) -> list:
        """Return articles mentioning any of the given team names within the last N hours."""
        pass

    @abstractmethod
    def get_by_game_id(self, game_id: int) -> list:
        """Return articles linked to a specific game."""
        pass
