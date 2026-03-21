from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_game_repo, get_prediction_repo, get_prediction_service
from app.api.schemas import PredictionResponse
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_predictions_repository import SqlAlchemyPredictionRepository
from app.services.prediction_service import PredictionService

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.post("/run")
def run_predictions(
    game_date: date = Query(description="Date of games to predict (YYYY-MM-DD)"),
    service: PredictionService = Depends(get_prediction_service),
    game_repo: GameRepository = Depends(get_game_repo),
):
    """
    Run totals + moneyline predictions for all games on the given date
    and persist the results.

    Requires that odds lines (hr_total_runs_line, home_ml_price, away_ml_price)
    have already been ingested via POST /odds/lines for the same date.
    """
    games = game_repo.get_games_by_date(game_date)
    if not games:
        raise HTTPException(status_code=404, detail=f"No games found for {game_date}")

    results = []
    for game in games:
        result = service.predict_and_log_for_game(game)
        results.append({"game_id": game.id, **result})

    return {"date": str(game_date), "games_processed": len(results), "results": results}


@router.post("/game/{game_id}")
def predict_game(
    game_id: int,
    service: PredictionService = Depends(get_prediction_service),
    game_repo: GameRepository = Depends(get_game_repo),
):
    """
    Run totals + moneyline predictions for a single game and persist the results.
    """
    game = game_repo.get_by_id(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")

    result = service.predict_and_log_for_game(game)
    return {"game_id": game_id, **result}


@router.get("", response_model=List[PredictionResponse])
def get_predictions(
    game_date: date = Query(description="Filter predictions by game date"),
    team: Optional[str] = Query(default=None, description="Filter by team name"),
    market: Optional[str] = Query(default=None, description="Filter by market: TOTAL or H2H"),
    exclude_no_bet: bool = Query(default=False, description="Exclude NO BET recommendations"),
    repo: SqlAlchemyPredictionRepository = Depends(get_prediction_repo),
):
    """
    Retrieve stored predictions for a given date with optional filters.
    """
    return repo.get_predictions(
        game_date=game_date,
        team=team,
        market=market,
        exclude_no_bet=exclude_no_bet,
    )
