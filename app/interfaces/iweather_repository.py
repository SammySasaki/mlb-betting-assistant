from abc import ABC, abstractmethod
from typing import Optional
from app.db_models import Weather

class IWeatherRepository(ABC):
    @abstractmethod
    def get_by_game_id(self, game_id: int) -> Optional[Weather]:
        """Fetch the Weather record for a given game_id."""
        pass