from datetime import date
from infra.db.init_db import SessionLocal
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_features_repository import SqlAlchemyFeaturesRepository
from app.implementations.sqlalchemy_player_repository import SQLAlchemyPlayerRepository
from app.implementations.sqlalchemy_weather_repository import WeatherRepository
from app.implementations.sqlalchemy_predictions_repository import SqlAlchemyPredictionRepository
from app.services.feature_extractor import FeatureExtractor
from app.services.prediction_service import PredictionService


if __name__ == "__main__":
    session = SessionLocal()

    # Repositories
    pgs_repo = SqlAlchemyPlayerGameStatsRepository(session)
    game_repo = GameRepository(session)
    feature_repo = SqlAlchemyFeaturesRepository(session)
    player_repo = SQLAlchemyPlayerRepository(session)
    weather_repo = WeatherRepository(session)
    predictions_repo = SqlAlchemyPredictionRepository(session)

    # Feature extractor + service
    extractor = FeatureExtractor(pgs_repo, game_repo, feature_repo, player_repo, weather_repo)
    prediction_service = PredictionService(extractor, predictions_repo, game_repo)

    # Get today’s games
    today = date.today()
    games = game_repo.get_games_by_date(today)

    # Run predictions for each game
    for game in games:
        prediction_service.predict_and_log_for_game(game)