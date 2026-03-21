import os
import requests
from datetime import date, datetime

from app.interfaces.iweather_client import IWeatherClient, WeatherReading, WeatherApiLimitError
from app.utils.cache import load as cache_load, save as cache_save

_BASE_URL = (
    "https://weather.visualcrossing.com"
    "/VisualCrossingWebServices/rest/services/timeline"
)

# Today's weather is refreshed every 2 hours; historical is cached permanently.
_TODAY_TTL = 7_200


class VisualCrossingWeatherClient(IWeatherClient):
    """Fetches hourly weather from the Visual Crossing Timeline API.

    Responses are cached to disk so repeated ingestion runs (e.g. backfills)
    do not burn daily API quota.  Historical dates are stored indefinitely;
    today's data expires after two hours.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if api_key is None:
            api_key = os.environ["VISUAL_CROSSING_API_KEY"]
        self._api_key = api_key
        self._session = requests.Session()

    def get_hourly_weather(
        self, city: str, target_date: date, hour: int
    ) -> WeatherReading | None:
        """Return a WeatherReading for the requested hour.

        Raises WeatherApiLimitError when the daily quota is exhausted so the
        caller can stop early and commit progress without a partial day.
        """
        cache_key = f"{city}_{target_date.isoformat()}"
        cached = cache_load("weather", cache_key, target_date=target_date, ttl=_TODAY_TTL)

        if cached is None:
            resp = self._session.get(
                f"{_BASE_URL}/{city}/{target_date.isoformat()}",
                params={"unitGroup": "us", "key": self._api_key, "include": "hours"},
            )

            if resp.status_code != 200:
                if resp.text.strip() == "Maximum daily cost exceeded":
                    raise WeatherApiLimitError("Visual Crossing daily quota exhausted")
                return None

            data = resp.json()
            if "days" not in data:
                return None

            cache_save("weather", cache_key, data)
            cached = data

        for hour_data in cached["days"][0]["hours"]:
            if datetime.strptime(hour_data["datetime"], "%H:%M:%S").hour == hour:
                return WeatherReading(
                    temperature=hour_data.get("temp"),
                    wind_speed=hour_data.get("windspeed"),
                    wind_direction=hour_data.get("winddir"),
                    precipitation=hour_data.get("precip"),
                )

        return None  # exact hour not in response
