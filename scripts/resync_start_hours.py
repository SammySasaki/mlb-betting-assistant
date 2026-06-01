"""
Re-sync start_hour_utc for all historical games from the MLB Stats API.

The ingestion service previously only set start_hour_utc when it was NULL,
so games ingested with a wrong/placeholder value were never corrected.
This script overwrites start_hour_utc for every game using the authoritative
gameDate field from the MLB API.

Usage:
    docker compose run --rm app python scripts/resync_start_hours.py
    docker compose run --rm app python scripts/resync_start_hours.py --season 2022
    docker compose run --rm app python scripts/resync_start_hours.py --dry-run
"""

import argparse
import json
import logging
import time
from datetime import datetime

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Game

CHANGED_IDS_FILE = "scripts/resync_changed_game_ids.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def get_schedule(target_date, session: requests.Session) -> list:
    resp = session.get(MLB_SCHEDULE_URL, params={
        "sportId": 1,
        "date": target_date.strftime("%Y-%m-%d"),
        "hydrate": "team",
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("dates", [])[0].get("games", []) if data.get("dates") else []


def parse_args():
    p = argparse.ArgumentParser(description="Re-sync start_hour_utc from MLB API")
    p.add_argument("--season", type=int, help="Only sync games from this season year")
    p.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    p.add_argument("--delay", type=float, default=0.3, help="Seconds between API calls")
    return p.parse_args()


def main():
    import os
    args = parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL not set")

    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)

    http = requests.Session()

    with Session() as session:
        query = session.query(Game)
        if args.season:
            query = query.filter(Game.season_year == args.season)
        else:
            query = query.filter(Game.season_year >= 2022)

        games = query.order_by(Game.date).all()
        log.info("Loaded %d games from DB", len(games))

        # Build lookup: game_id -> db game
        by_id = {g.id: g for g in games}

        # Group by date to minimize API calls (one call per date)
        dates = sorted({g.date for g in games})
        log.info("Spanning %d unique dates", len(dates))

        updated = 0
        unchanged = 0
        not_found = 0
        changed_ids = []  # game IDs whose start_hour_utc changed

        for d in dates:
            try:
                api_games = get_schedule(d, http)
            except Exception as exc:
                log.warning("  %s — API error: %s", d, exc)
                time.sleep(args.delay)
                continue

            for g_data in api_games:
                game_id = g_data.get("gamePk")
                if game_id not in by_id:
                    continue

                raw_date = g_data.get("gameDate", "")
                if not raw_date:
                    continue

                dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                new_hour = dt.hour

                db_game = by_id[game_id]
                old_hour = db_game.start_hour_utc

                if old_hour != new_hour:
                    log.info("  %s game %d (%s @ %s): %s → %s",
                             d, game_id, db_game.away_team, db_game.home_team,
                             old_hour, new_hour)
                    if not args.dry_run:
                        db_game.start_hour_utc = new_hour
                    changed_ids.append(game_id)
                    updated += 1
                else:
                    unchanged += 1

            time.sleep(args.delay)

        log.info("Done. Updated: %d  Unchanged: %d  Not in DB: %d",
                 updated, unchanged, not_found)

        if not args.dry_run and updated > 0:
            session.commit()
            log.info("Committed %d updates.", updated)
            with open(CHANGED_IDS_FILE, "w") as f:
                json.dump(changed_ids, f)
            log.info("Saved %d changed game IDs to %s", len(changed_ids), CHANGED_IDS_FILE)
        elif args.dry_run:
            log.info("Dry run — no changes written.")
            log.info("Would change game IDs: %s", changed_ids)


if __name__ == "__main__":
    main()
