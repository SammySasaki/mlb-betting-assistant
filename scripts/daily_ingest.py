"""
scripts/daily_ingest.py

Daily MLB ingestion job — run via Railway Cron at 11am ET (16:00 UTC).

Steps (in order):
  1. Ingest upcoming games + lineups for today
  2. Ingest current odds lines for today
  3. Backfill yesterday's completed game data (scores, box scores, features)
  4. Run predictions for today's games
"""
import logging
import sys
from datetime import date, timedelta

from infra.db.init_db import SessionLocal
from app.services.ingestion_service import MLBIngestionService
from app.services.odds_service import OddsService
from app.services.feature_extractor import FeatureExtractor
from app.services.prediction_service import PredictionService
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_predictions_repository import SqlAlchemyPredictionRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.sqlalchemy_features_repository import SqlAlchemyFeaturesRepository
from app.implementations.sqlalchemy_player_repository import SQLAlchemyPlayerRepository
from app.implementations.sqlalchemy_weather_repository import WeatherRepository

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

today = date.today()
yesterday = today - timedelta(days=1)
errors = []

# ---------------------------------------------------------------------------
# 1. Upcoming games
# ---------------------------------------------------------------------------
log.info("Step 1/4 - ingest upcoming games for %s", today)
try:
    summary = MLBIngestionService().ingest_upcoming_day(today)
    log.info("  %d game(s) stored", summary["games_stored"])
    if summary["errors"]:
        errors.extend(summary["errors"])
except Exception as exc:
    log.exception("  FAILED: %s", exc)
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 2. Odds
# ---------------------------------------------------------------------------
log.info("Step 2/4 - ingest odds for %s", today)
try:
    result = OddsService().ingest_lines_for_date(today)
    log.info("  totals: %d  ML: %d", result["totals_updated"], result["ml_updated"])
    if result.get("errors"):
        errors.extend(result["errors"])
except Exception as exc:
    log.exception("  FAILED: %s", exc)
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 3. Backfill yesterday
# ---------------------------------------------------------------------------
log.info("Step 3/4 - backfill completed games for %s", yesterday)
try:
    summary = MLBIngestionService().ingest_dates(yesterday, yesterday)
    log.info("  %d game(s) processed", summary["total_games"])
    if summary["total_errors"]:
        errors.append(f"backfill had {summary['total_errors']} error(s)")
except Exception as exc:
    log.exception("  FAILED: %s", exc)
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# 4. Predictions
# ---------------------------------------------------------------------------
log.info("Step 4/4 - run predictions for %s", today)
try:
    session = SessionLocal()
    game_repo = GameRepository(session)
    extractor = FeatureExtractor(
        SqlAlchemyPlayerGameStatsRepository(session),
        game_repo,
        SqlAlchemyFeaturesRepository(session),
        SQLAlchemyPlayerRepository(session),
        WeatherRepository(session),
    )
    service = PredictionService(extractor, SqlAlchemyPredictionRepository(session), game_repo)
    games = game_repo.get_games_by_date(today)
    log.info("  %d game(s) to predict", len(games))
    for game in games:
        service.predict_and_log_for_game(game)
    session.close()
except Exception as exc:
    log.exception("  FAILED: %s", exc)
    errors.append(str(exc))

# ---------------------------------------------------------------------------
# Exit
# ---------------------------------------------------------------------------
if errors:
    log.error("Job completed with %d error(s):", len(errors))
    for e in errors:
        log.error("  - %s", e)
    sys.exit(1)

log.info("Daily ingest complete.")
