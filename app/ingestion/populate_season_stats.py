"""
Backfill player_season_stats for a full season.

Usage:
    docker-compose run --rm app python app/ingestion/populate_season_stats.py 2024
    docker-compose run --rm app python app/ingestion/populate_season_stats.py      # defaults to 2024
"""

import sys
from infra.db.init_db import SessionLocal
from app.implementations.sqlalchemy_player_season_stats_repository import SQLAlchemyPlayerSeasonStatsRepository
from app.services.season_stats_service import SeasonStatsService


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

    with SessionLocal() as session:
        repo = SQLAlchemyPlayerSeasonStatsRepository(session)
        service = SeasonStatsService(session, repo)
        result = service.calculate_season_stats(season)

    print(result)
