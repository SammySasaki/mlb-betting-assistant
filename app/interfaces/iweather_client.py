from abc import ABC, abstractmethod
from datetime import date
from typing import TypedDict


class WeatherReading(TypedDict):
    """Normalised weather snapshot for a single hour at a location."""
    temperature: float | None    # degrees Fahrenheit
    wind_speed: float | None     # mph
    wind_direction: float | None # degrees (0–360)
    precipitation: float | None  # inches


class WeatherApiLimitError(Exception):
    """Raised when the weather provider's daily request quota is exhausted."""


class IWeatherClient(ABC):
    """Interface for fetching hourly weather data from any weather provider.

    To swap providers (e.g. Visual Crossing → OpenWeather) implement this
    interface and inject the new class – nothing else needs to change.
    """

    @abstractmethod
    def get_hourly_weather(
        self, city: str, target_date: date, hour: int
    ) -> WeatherReading | None:
        """Return a WeatherReading for the closest available hour.

        Parameters
        ----------
        city        : Human-readable location string (e.g. "Boston, MA").
        target_date : Calendar date of the game.
        hour        : Local hour of day (0–23) to look up.

        Returns
        -------
        WeatherReading if data is available, None if the location or hour
        is not found.

        Raises
        ------
        WeatherApiLimitError
            If the provider's daily credit/request limit is exhausted.
        """
        pass
