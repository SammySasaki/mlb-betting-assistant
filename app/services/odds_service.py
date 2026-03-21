"""
app/services/odds_service.py

Service layer for betting-odds ingestion.

Public API
----------
    service = OddsService()                           # uses TheOddsApiClient by default
    service = OddsService(odds_client=mock_client)    # injectable for testing

    service.ingest_lines_for_date(date.today())            # current totals + ML
    service.ingest_historical_lines_for_date(past_date)    # historical totals only
"""

import logging
import os
from datetime import date
from difflib import SequenceMatcher

from sqlalchemy import text

from infra.db.init_db import SessionLocal
from app.db.models import Game
from app.interfaces.iodds_client import IOddsClient
from app.implementations.the_odds_client import TheOddsApiClient

logger = logging.getLogger(__name__)


class OddsService:
    """Matches odds from an external provider to Game rows and persists the lines."""

    def __init__(
        self,
        odds_client: IOddsClient | None = None,
        bookmaker: str = "hardrockbet",
    ) -> None:
        self._client = odds_client or TheOddsApiClient(
            api_key=os.environ["THE_ODDS_API_KEY"],
            bookmaker=bookmaker,
        )
        self._bookmaker = bookmaker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_lines_for_date(self, target_date: date) -> dict:
        """Fetch and store current totals + moneyline odds for games on target_date.

        Intended for upcoming games (today or tomorrow).  Scores are not required.

        Returns
        -------
        dict with keys:
            date           – ISO date string
            totals_updated – number of games that received totals lines
            ml_updated     – number of games that received ML prices
            errors         – list of error message strings
        """
        summary = {
            "date": str(target_date),
            "totals_updated": 0,
            "ml_updated": 0,
            "errors": [],
        }

        totals_odds = self._client.get_totals_odds()
        ml_odds = self._client.get_ml_odds()

        with SessionLocal() as session:
            games = session.query(Game).filter(Game.date == target_date).all()

            for game in games:
                try:
                    match = self._find_match(game, totals_odds)
                    if match and self._apply_totals(game, match):
                        summary["totals_updated"] += 1

                    match = self._find_match(game, ml_odds)
                    if match and self._apply_moneylines(game, match):
                        summary["ml_updated"] += 1
                except Exception as exc:
                    session.rollback()
                    msg = f"Game {game.id} ({game.away_team} @ {game.home_team}): {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)

            session.commit()

        logger.info(
            "ingest_lines_for_date %s – totals: %d, ML: %d, errors: %d",
            target_date, summary["totals_updated"], summary["ml_updated"], len(summary["errors"]),
        )
        return summary

    def ingest_historical_lines_for_date(self, target_date: date) -> dict:
        """Fetch and store historical totals lines for a past date.

        Returns
        -------
        dict with keys:
            date    – ISO date string
            updated – number of games that received totals lines
            errors  – list of error message strings
        """
        summary = {"date": str(target_date), "updated": 0, "errors": []}

        odds_data = self._client.get_historical_totals_odds(target_date)

        with SessionLocal() as session:
            games = session.query(Game).filter(Game.date == target_date).all()

            for game in games:
                try:
                    match = self._find_match(game, odds_data)
                    if match and self._apply_totals(game, match):
                        summary["updated"] += 1
                except Exception as exc:
                    session.rollback()
                    msg = f"Game {game.id} ({game.away_team} @ {game.home_team}): {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)

            session.commit()

        logger.info(
            "ingest_historical_lines_for_date %s – updated: %d, errors: %d",
            target_date, summary["updated"], len(summary["errors"]),
        )
        return summary

    # ------------------------------------------------------------------
    # Private – matching + DB update helpers
    # ------------------------------------------------------------------

    def _find_match(self, game: Game, odds_data: list) -> dict | None:
        """Return the odds entry whose teams fuzzy-match the given game, or None."""
        for entry in odds_data:
            if (
                self._names_match(game.home_team, entry["home_team"])
                and self._names_match(game.away_team, entry["away_team"])
            ):
                return entry
        return None

    def _apply_totals(self, game: Game, odds_entry: dict) -> bool:
        """Write over/under line and prices onto the game. Returns True if applied."""
        bookmaker = self._find_bookmaker(odds_entry)
        if not bookmaker:
            return False

        market = next((m for m in bookmaker["markets"] if m["key"] == "totals"), None)
        if not market:
            return False

        over_price = under_price = total_line = None
        for outcome in market["outcomes"]:
            if outcome["name"] == "Over":
                total_line = outcome["point"]
                over_price = outcome["price"]
            elif outcome["name"] == "Under":
                under_price = outcome["price"]

        if None in (total_line, over_price, under_price):
            return False

        game.hr_total_runs_line = total_line
        game.over_price = over_price
        game.under_price = under_price
        logger.debug("Totals applied to game %s: line=%s", game.id, total_line)
        return True

    def _apply_moneylines(self, game: Game, odds_entry: dict) -> bool:
        """Write home/away ML prices onto the game. Returns True if applied."""
        bookmaker = self._find_bookmaker(odds_entry)
        if not bookmaker:
            return False

        market = next((m for m in bookmaker["markets"] if m["key"] == "h2h"), None)
        if not market:
            return False

        home_price = away_price = None
        for outcome in market["outcomes"]:
            if self._names_match(outcome["name"], odds_entry["home_team"]):
                home_price = outcome["price"]
            elif self._names_match(outcome["name"], odds_entry["away_team"]):
                away_price = outcome["price"]

        if home_price is None or away_price is None:
            return False

        game.home_ml_price = home_price
        game.away_ml_price = away_price
        logger.debug("ML applied to game %s: home=%s away=%s", game.id, home_price, away_price)
        return True

    def _find_bookmaker(self, odds_entry: dict) -> dict | None:
        return next(
            (b for b in odds_entry.get("bookmakers", []) if b["key"] == self._bookmaker),
            None,
        )

    @staticmethod
    def _names_match(name1: str, name2: str, threshold: float = 0.8) -> bool:
        """Fuzzy team-name comparison (handles 'Athletics' vs 'Oakland Athletics')."""
        n1, n2 = name1.lower(), name2.lower()
        if n1 == n2:
            return True
        if SequenceMatcher(None, n1, n2).ratio() >= threshold:
            return True
        return n1 in n2 or n2 in n1
