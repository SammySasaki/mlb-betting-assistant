from typing import Generator
from fastapi import Depends
from sqlalchemy.orm import Session

from infra.db.init_db import SessionLocal
from app.services.ingestion_service import MLBIngestionService
from app.services.odds_service import OddsService
from app.services.prediction_service import PredictionService
from app.services.lineup_service import LineupService
from app.services.feature_extractor import FeatureExtractor
from app.services.at_bat_service import AtBatService
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_predictions_repository import SqlAlchemyPredictionRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.sqlalchemy_features_repository import SqlAlchemyFeaturesRepository
from app.implementations.sqlalchemy_player_repository import SQLAlchemyPlayerRepository
from app.implementations.sqlalchemy_weather_repository import WeatherRepository
from app.implementations.sqlalchemy_lineup_repository import SQLAlchemyLineupRepository
from app.implementations.mlb_api_client import StatsApiClient
from app.implementations.sqlalchemy_player_season_stats_repository import SQLAlchemyPlayerSeasonStatsRepository
from app.services.season_stats_service import SeasonStatsService
from app.services.news_service import NewsService
from app.implementations.go_news_client import GoNewsClient
from app.implementations.sqlalchemy_news_repository import SqlAlchemyNewsRepository


def get_db() -> Generator[Session, None, None]:
    with SessionLocal() as session:
        yield session


# ---------------------------------------------------------------------------
# Simple service factories (manage their own sessions internally)
# ---------------------------------------------------------------------------

def get_ingestion_service() -> MLBIngestionService:
    return MLBIngestionService()


def get_odds_service() -> OddsService:
    return OddsService()


def get_at_bat_service() -> AtBatService:
    return AtBatService()


# ---------------------------------------------------------------------------
# Session-scoped service factories
# ---------------------------------------------------------------------------

def get_game_repo(session: Session = Depends(get_db)) -> GameRepository:
    return GameRepository(session)


def get_prediction_repo(
    session: Session = Depends(get_db),
) -> SqlAlchemyPredictionRepository:
    return SqlAlchemyPredictionRepository(session)


def get_prediction_service(session: Session = Depends(get_db)) -> PredictionService:
    pgs_repo = SqlAlchemyPlayerGameStatsRepository(session)
    game_repo = GameRepository(session)
    feature_repo = SqlAlchemyFeaturesRepository(session)
    player_repo = SQLAlchemyPlayerRepository(session)
    weather_repo = WeatherRepository(session)
    extractor = FeatureExtractor(pgs_repo, game_repo, feature_repo, player_repo, weather_repo)
    prediction_repo = SqlAlchemyPredictionRepository(session)
    return PredictionService(extractor, prediction_repo, game_repo)


def get_lineup_service(session: Session = Depends(get_db)) -> LineupService:
    pgs_repo = SqlAlchemyPlayerGameStatsRepository(session)
    lineup_repo = SQLAlchemyLineupRepository(session)
    game_repo = GameRepository(session)
    player_repo = SQLAlchemyPlayerRepository(session)
    mlb_client = StatsApiClient()
    return LineupService(pgs_repo, lineup_repo, mlb_client, game_repo, player_repo)


def get_season_stats_repo(
    session: Session = Depends(get_db),
) -> SQLAlchemyPlayerSeasonStatsRepository:
    return SQLAlchemyPlayerSeasonStatsRepository(session)


def get_season_stats_service(session: Session = Depends(get_db)) -> SeasonStatsService:
    repo = SQLAlchemyPlayerSeasonStatsRepository(session)
    return SeasonStatsService(session, repo)


def get_news_service(session: Session = Depends(get_db)) -> NewsService:
    client = GoNewsClient()
    repo = SqlAlchemyNewsRepository(session)
    return NewsService(client, repo, session)
