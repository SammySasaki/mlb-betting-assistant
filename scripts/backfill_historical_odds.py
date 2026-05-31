"""
scripts/backfill_historical_odds.py

Backfill game-time (closing-line) odds for historical games using the
paid historical endpoint from The Odds API.

Strategy
--------
Games are grouped by (date, start_hour_utc).  One API call is made per
unique start-time slot — typically 2-4 calls per calendar date — returning
a snapshot of all active odds at that moment.  Snapping at start_hour+5min
captures the closing line after the market has locked.

Requirements
------------
  - The Odds API paid historical plan (data back to 2020)
  - THE_ODDS_API_KEY environment variable

Usage
-----
    # Backfill all missing odds for 2021-2024 (default):
    docker compose run --rm app python scripts/backfill_historical_odds.py

    # Single season:
    docker compose run --rm app python scripts/backfill_historical_odds.py --season 2023

    # Date range:
    docker compose run --rm app python scripts/backfill_historical_odds.py --start 2023-04-01 --end 2023-10-31

    # Preview what would be fetched without touching DB or API:
    docker compose run --rm app python scripts/backfill_historical_odds.py --dry-run

    # Re-fetch even where odds already exist (overwrite):
    docker compose run --rm app python scripts/backfill_historical_odds.py --overwrite --season 2024

Credit estimate (The Odds API)
-------------------------------
Each historical call costs: num_events × num_markets × num_regions credits.
  ~12 events/day × 2 markets × 1 region = ~24 credits/slot
  ~4 slots/day × 180 game-days/season × 4 seasons ≈ 2,880 slots → ~70k credits

Check remaining credits after each run — the script prints them per call.
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from difflib import SequenceMatcher

import requests
from sqlalchemy import or_

from infra.db.init_db import SessionLocal
from app.db.models import Game

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SPORT = "baseball_mlb"
REGION = "us"
DEFAULT_BOOKMAKER = "hardrockbet"
# Fetched alongside the primary in every call at no extra credit cost.
# hardrockbet is preferred; draftkings fills in when hardrockbet has no line.
FALLBACK_BOOKMAKER = "draftkings"
BASE_URL = "https://api.the-odds-api.com/v4"
# Both markets in one call — halves credit usage vs two separate calls.
MARKETS = "totals,h2h"
# Minutes past the start hour to snapshot — lines are final by then.
SNAPSHOT_OFFSET_MINUTES = 5


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill historical game-time (closing-line) odds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--season", type=int,
                   help="Single season year, e.g. 2023")
    p.add_argument("--start", type=date.fromisoformat, metavar="YYYY-MM-DD",
                   help="Start date inclusive")
    p.add_argument("--end", type=date.fromisoformat, metavar="YYYY-MM-DD",
                   help="End date inclusive")
    p.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER,
                   help=f"Primary bookmaker key (default: {DEFAULT_BOOKMAKER})")
    p.add_argument("--fallback", default=FALLBACK_BOOKMAKER,
                   help=f"Fallback bookmaker when primary has no line (default: {FALLBACK_BOOKMAKER})")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing odds (default: skip games that already have odds)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print plan without making any API calls or DB writes")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds to sleep between API calls (default: 0.5)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def fetch_historical_odds(
    api_key: str,
    snapshot_utc: datetime,
    bookmakers: list[str],
) -> list[dict]:
    """Call the paid historical endpoint and return the data array.

    Passing multiple bookmaker keys in one call costs the same credits as one
    bookmaker — the API charges per event/market/region, not per bookmaker.

    Response envelope:
        {"timestamp": "...", "previous_timestamp": "...", "next_timestamp": "...", "data": [...]}
    """
    resp = requests.get(
        f"{BASE_URL}/historical/sports/{SPORT}/odds",
        params={
            "regions": REGION,
            "markets": MARKETS,
            "bookmakers": ",".join(bookmakers),
            "apiKey": api_key,
            "date": snapshot_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    resp.raise_for_status()

    remaining = resp.headers.get("x-requests-remaining", "?")
    used = resp.headers.get("x-requests-used", "?")
    log.info("  credits used: %s  remaining: %s", used, remaining)

    payload = resp.json()
    # Historical endpoint may wrap in an envelope or return a bare list.
    if isinstance(payload, dict):
        return payload.get("data", [])
    return payload


# ---------------------------------------------------------------------------
# Matching + applying
# ---------------------------------------------------------------------------

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def names_match(name1: str, name2: str, threshold: float = 0.75) -> bool:
    n1, n2 = name1.lower().strip(), name2.lower().strip()
    if n1 == n2:
        return True
    if _sim(n1, n2) >= threshold:
        return True
    return n1 in n2 or n2 in n1


def find_bookmaker_data(odds_entry: dict, bookmakers: list[str]) -> tuple[dict | None, str | None]:
    """Return the first bookmaker block found from the priority list, and its key."""
    for key in bookmakers:
        bk = next((b for b in odds_entry.get("bookmakers", []) if b["key"] == key), None)
        if bk is not None:
            return bk, key
    return None, None


def apply_totals(game: Game, odds_entry: dict, bookmakers: list[str]) -> tuple[bool, str | None]:
    """Write O/U line and prices onto game using the first available bookmaker.

    Returns (applied: bool, bookmaker_used: str | None).
    """
    bk, bk_key = find_bookmaker_data(odds_entry, bookmakers)
    if not bk:
        return False, None
    market = next((m for m in bk["markets"] if m["key"] == "totals"), None)
    if not market:
        return False, None

    total_line = over_price = under_price = None
    for outcome in market["outcomes"]:
        if outcome["name"] == "Over":
            total_line = outcome["point"]
            over_price = outcome["price"]
        elif outcome["name"] == "Under":
            under_price = outcome["price"]

    if None in (total_line, over_price, under_price):
        return False, None

    game.hr_total_runs_line = total_line
    game.over_price = over_price
    game.under_price = under_price
    return True, bk_key


def apply_moneylines(game: Game, odds_entry: dict, bookmakers: list[str]) -> tuple[bool, str | None]:
    """Write home/away ML prices onto game using the first available bookmaker.

    Returns (applied: bool, bookmaker_used: str | None).
    """
    bk, bk_key = find_bookmaker_data(odds_entry, bookmakers)
    if not bk:
        return False, None
    market = next((m for m in bk["markets"] if m["key"] == "h2h"), None)
    if not market:
        return False, None

    home_price = away_price = None
    for outcome in market["outcomes"]:
        if names_match(outcome["name"], odds_entry["home_team"]):
            home_price = outcome["price"]
        elif names_match(outcome["name"], odds_entry["away_team"]):
            away_price = outcome["price"]

    if home_price is None or away_price is None:
        return False, None

    game.home_ml_price = home_price
    game.away_ml_price = away_price
    return True, bk_key


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    api_key = os.environ.get("THE_ODDS_API_KEY", "")
    if not api_key and not args.dry_run:
        log.error("THE_ODDS_API_KEY environment variable not set")
        sys.exit(1)

    with SessionLocal() as session:
        query = session.query(Game)

        # Scope filter
        if args.season:
            query = query.filter(Game.season_year == args.season)
        elif args.start and args.end:
            query = query.filter(Game.date >= args.start, Game.date <= args.end)
        else:
            # Default: all historical seasons
            query = query.filter(Game.season_year.between(2021, 2024))

        # Skip games that already have complete odds unless --overwrite
        if not args.overwrite:
            query = query.filter(
                or_(
                    Game.hr_total_runs_line.is_(None),
                    Game.home_ml_price.is_(None),
                    Game.away_ml_price.is_(None),
                )
            )

        games = query.order_by(Game.date, Game.start_hour_utc).all()

    if not games:
        log.info("No games found matching criteria — nothing to backfill.")
        return

    log.info("Found %d games to backfill.", len(games))

    # Group by (date, start_hour) — one API call per unique slot.
    # Games with NULL start_hour_utc default to 18 UTC (1 pm ET / typical day game).
    slots: dict[tuple[date, int], list[Game]] = defaultdict(list)
    for game in games:
        hour = game.start_hour_utc if game.start_hour_utc is not None else 18
        slots[(game.date, hour)].append(game)

    log.info("Grouped into %d unique start-time slots (~%d API calls).",
             len(slots), len(slots))

    if args.dry_run:
        log.info("--- DRY RUN — no API calls or DB writes ---")
        for (d, h), gs in sorted(slots.items()):
            matchups = ", ".join(f"{g.away_team}@{g.home_team}" for g in gs)
            log.info("  %s %02d:05 UTC — %d game(s): %s", d, h, len(gs), matchups)
        return

    # Priority-ordered bookmaker list: primary first, fallback second.
    # Both are requested in one API call; primary is preferred per game.
    bookmakers = [args.bookmaker]
    if args.fallback and args.fallback != args.bookmaker:
        bookmakers.append(args.fallback)

    log.info("Bookmaker priority: %s", " → ".join(bookmakers))

    # Process slots
    stats = {"totals": 0, "ml": 0, "no_match": 0, "api_errors": 0, "fallback_used": 0}

    with SessionLocal() as session:
        # Re-attach games to this session for writing
        game_ids = {g.id for g in games}

        for (game_date, start_hour), slot_games in sorted(slots.items()):
            snapshot_utc = datetime(
                game_date.year, game_date.month, game_date.day,
                start_hour, SNAPSHOT_OFFSET_MINUTES, 0,
                tzinfo=timezone.utc,
            )

            log.info("Fetching %s %02d:%02d UTC (%d games)...",
                     game_date, start_hour, SNAPSHOT_OFFSET_MINUTES, len(slot_games))

            try:
                odds_data = fetch_historical_odds(api_key, snapshot_utc, bookmakers)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else "?"
                log.error("  HTTP %s — skipping slot %s %02d:00", code, game_date, start_hour)
                stats["api_errors"] += len(slot_games)
                time.sleep(args.delay)
                continue
            except Exception as exc:
                log.error("  Error fetching %s %02d:00: %s", game_date, start_hour, exc)
                stats["api_errors"] += len(slot_games)
                time.sleep(args.delay)
                continue

            if not odds_data:
                log.warning("  Empty response for %s %02d:00 — bookmaker may lack coverage for this date",
                            game_date, start_hour)

            # Re-fetch session-bound game objects so SQLAlchemy can track changes
            db_games = {
                g.id: g
                for g in session.query(Game).filter(Game.id.in_([g.id for g in slot_games])).all()
            }

            for game in slot_games:
                db_game = db_games.get(game.id)
                if db_game is None:
                    continue

                matched = next(
                    (e for e in odds_data
                     if names_match(db_game.home_team, e.get("home_team", ""))
                     and names_match(db_game.away_team, e.get("away_team", ""))),
                    None,
                )

                if matched is None:
                    log.warning("  No match: %s @ %s (%s)",
                                db_game.away_team, db_game.home_team, game_date)
                    stats["no_match"] += 1
                    continue

                needs_totals = args.overwrite or db_game.hr_total_runs_line is None
                needs_ml = args.overwrite or (
                    db_game.home_ml_price is None or db_game.away_ml_price is None
                )

                if needs_totals:
                    applied, bk_used = apply_totals(db_game, matched, bookmakers)
                    if applied:
                        stats["totals"] += 1
                        if bk_used != bookmakers[0]:
                            stats["fallback_used"] += 1
                            log.debug("  totals (fallback %s): %s@%s line=%.1f",
                                      bk_used, db_game.away_team, db_game.home_team,
                                      db_game.hr_total_runs_line)
                        else:
                            log.debug("  totals: %s@%s line=%.1f",
                                      db_game.away_team, db_game.home_team,
                                      db_game.hr_total_runs_line)

                if needs_ml:
                    applied, bk_used = apply_moneylines(db_game, matched, bookmakers)
                    if applied:
                        stats["ml"] += 1
                        if bk_used != bookmakers[0]:
                            stats["fallback_used"] += 1
                            log.debug("  ML (fallback %s): %s@%s home=%.3f away=%.3f",
                                      bk_used, db_game.away_team, db_game.home_team,
                                      db_game.home_ml_price, db_game.away_ml_price)
                        else:
                            log.debug("  ML: %s@%s home=%.3f away=%.3f",
                                      db_game.away_team, db_game.home_team,
                                      db_game.home_ml_price, db_game.away_ml_price)

            session.commit()
            log.info("  Slot committed. Cumulative — totals: %(totals)d  ML: %(ml)d  "
                     "fallback: %(fallback_used)d  no_match: %(no_match)d  errors: %(api_errors)d", stats)
            time.sleep(args.delay)

    log.info(
        "Backfill complete.\n"
        "  Totals updated : %d\n"
        "  ML updated     : %d\n"
        "  Fallback used  : %d\n"
        "  No match       : %d\n"
        "  API errors     : %d",
        stats["totals"], stats["ml"], stats["fallback_used"],
        stats["no_match"], stats["api_errors"],
    )

    if stats["no_match"] > 0:
        log.info(
            "Tip: remaining unmatched games mean neither bookmaker had a line. "
            "Re-run with --fallback fanduel (or bovada) to try a different source."
        )


if __name__ == "__main__":
    main()
