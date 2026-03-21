import time
import requests
from datetime import date

from app.interfaces.iodds_client import IOddsClient
from app.utils.cache import load as cache_load, save as cache_save

_DEFAULT_SPORT = "baseball_mlb"
_DEFAULT_REGION = "us"
_DEFAULT_BOOKMAKER = "hardrockbet"

# Live odds are cached in-memory for 30 minutes to avoid duplicate API calls
# within a single session (e.g. fetching totals then ML odds back-to-back).
_LIVE_TTL = 1_800


class TheOddsApiClient(IOddsClient):
    """Fetches betting odds from The Odds API (https://the-odds-api.com).

    Caching strategy:
    - Historical odds (by date): permanent disk cache — past snapshots never change.
    - Live current odds: in-memory cache with a 30-minute TTL per market.
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(
        self,
        api_key: str,
        sport: str = _DEFAULT_SPORT,
        region: str = _DEFAULT_REGION,
        bookmaker: str = _DEFAULT_BOOKMAKER,
    ) -> None:
        self._api_key = api_key
        self._sport = sport
        self._region = region
        self.bookmaker = bookmaker  # public so OddsService can reference it
        self._session = requests.Session()
        # {market: (data, fetched_at_epoch)}
        self._live_cache: dict[str, tuple[list, float]] = {}

    def get_totals_odds(self) -> list:
        """Return current over/under odds for all upcoming games."""
        return self._fetch_odds_live(market="totals")

    def get_ml_odds(self) -> list:
        """Return current moneyline odds for all upcoming games."""
        return self._fetch_odds_live(market="h2h")

    def get_historical_totals_odds(self, target_date: date) -> list:
        """Return historical totals odds snapshotted at midnight on target_date."""
        cache_key = f"{self._sport}_{self.bookmaker}_{target_date.isoformat()}"
        cached = cache_load("odds_history", cache_key, target_date=target_date)
        if cached is not None:
            return cached

        resp = self._session.get(
            f"{self.BASE_URL}/sports/{self._sport}/odds-history",
            params={
                "regions": self._region,
                "markets": "totals",
                "bookmakers": self.bookmaker,
                "apiKey": self._api_key,
                "date": f"{target_date.isoformat()}T00:00:00Z",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        cache_save("odds_history", cache_key, data)
        return data

    def _fetch_odds_live(self, market: str) -> list:
        """Fetch live odds with a short in-memory TTL."""
        cached_data, fetched_at = self._live_cache.get(market, (None, 0.0))
        if cached_data is not None and time.time() - fetched_at < _LIVE_TTL:
            return cached_data

        resp = self._session.get(
            f"{self.BASE_URL}/sports/{self._sport}/odds",
            params={
                "regions": self._region,
                "markets": market,
                "bookmakers": self.bookmaker,
                "apiKey": self._api_key,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        self._live_cache[market] = (data, time.time())
        return data
