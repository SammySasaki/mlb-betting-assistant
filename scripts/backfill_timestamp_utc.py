"""
Bulk-populate games.timestamp_utc from the MLB Stats API gameDate field.

gameDate is the authoritative UTC first-pitch time. officialDate (stored in
games.date) is the local US calendar date, which differs from the UTC date for
late-evening West Coast games that cross midnight UTC.

Usage:
    docker compose run --rm app python scripts/backfill_timestamp_utc.py
    docker compose run --rm app python scripts/backfill_timestamp_utc.py --dry-run
    docker compose run --rm app python scripts/backfill_timestamp_utc.py --season 2024
"""

import argparse
import logging
import os
import time
from datetime import datetime, timezone

import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Game

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"


def get_schedule(target_date, http: requests.Session) -> list:
    resp = http.get(MLB_SCHEDULE_URL, params={
        "sportId": 1,
        "date": target_date.strftime("%Y-%m-%d"),
        "hydrate": "team",
    }, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    return data.get("dates", [])[0].get("games", []) if data.get("dates") else []


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delay", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    db_url = os.environ.get("DATABASE_URL")
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    http = requests.Session()

    with Session() as session:
        query = session.query(Game)
        if args.season:
            query = query.filter(Game.season_year == args.season)
        games = query.order_by(Game.date).all()
        log.info("Loaded %d games", len(games))

        by_id = {g.id: g for g in games}
        dates = sorted({g.date for g in games})
        log.info("Spanning %d unique dates", len(dates))

        updated = 0
        unchanged = 0

        for d in dates:
            try:
                api_games = get_schedule(d, http)
            except Exception as e:
                log.warning("  %s — API error: %s", d, e)
                time.sleep(args.delay)
                continue

            for g_data in api_games:
                game_id = g_data.get("gamePk")
                if game_id not in by_id:
                    continue
                raw = g_data.get("gameDate", "")
                if not raw:
                    continue
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                db_game = by_id[game_id]

                if db_game.timestamp_utc == dt:
                    unchanged += 1
                    continue

                log.info("  %s game %d (%s @ %s): %s → %s",
                         d, game_id, db_game.away_team, db_game.home_team,
                         db_game.timestamp_utc, dt)
                if not args.dry_run:
                    db_game.timestamp_utc = dt
                    db_game.start_hour_utc = dt.hour
                updated += 1

            time.sleep(args.delay)

        log.info("Done. Updated: %d  Unchanged: %d", updated, unchanged)

        if not args.dry_run and updated > 0:
            session.commit()
            log.info("Committed.")
        elif args.dry_run:
            log.info("Dry run — nothing written.")


if __name__ == "__main__":
    main()
