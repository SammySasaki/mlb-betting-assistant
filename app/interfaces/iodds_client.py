from abc import ABC, abstractmethod
from datetime import date


class IOddsClient(ABC):
    """Interface for fetching betting odds from any odds provider.

    To swap providers (e.g. The Odds API → Pinnacle) implement this interface
    and inject the new class into OddsService – nothing else needs to change.
    """

    @abstractmethod
    def get_totals_odds(self) -> list:
        """Return current over/under odds for all upcoming games.

        Each entry is a raw game dict from the provider containing at minimum:
            home_team, away_team, bookmakers[].markets[key="totals"].outcomes
        """
        pass

    @abstractmethod
    def get_ml_odds(self) -> list:
        """Return current moneyline (h2h) odds for all upcoming games.

        Each entry contains:
            home_team, away_team, bookmakers[].markets[key="h2h"].outcomes
        """
        pass

    @abstractmethod
    def get_historical_totals_odds(self, target_date: date) -> list:
        """Return historical totals odds snapshotted around target_date.

        Useful for backfilling lines for already-played games.
        """
        pass
