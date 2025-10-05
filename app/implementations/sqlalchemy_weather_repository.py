from sqlalchemy.orm import Session
from typing import Optional
from app.db_models import Weather
from app.interfaces.iweather_repository import IWeatherRepository

class WeatherRepository(IWeatherRepository):
    def __init__(self, session: Session):
        self.session = session

    def get_by_game_id(self, game_id: int) -> Optional[Weather]:
        return (
            self.session.query(Weather)
            .filter(Weather.game_id == game_id)
            .first()
        )