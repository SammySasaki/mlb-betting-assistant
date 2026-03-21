from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_lineup_service
from app.api.schemas import LineupDateRequest
from app.services.lineup_service import LineupService

router = APIRouter(prefix="/lineups", tags=["Lineups"])


@router.post("/game/{game_id}")
def fetch_lineups_for_game(
    game_id: int,
    service: LineupService = Depends(get_lineup_service),
):
    """
    Fetch and persist the lineup for a specific game from the MLB Stats API.

    Enriches each batter entry with season-to-date and last-10-games stats.
    Also updates probable pitcher IDs on the game record.
    """
    service.save_lineups_for_game(game_id)
    return {"game_id": game_id, "status": "ok"}


@router.post("/date")
def fetch_lineups_for_date(
    body: LineupDateRequest,
    service: LineupService = Depends(get_lineup_service),
):
    """
    Fetch and persist lineups for all games on the given date.
    """
    service.save_lineups_by_date(body.date)
    return {"date": str(body.date), "status": "ok"}


@router.post("/pitchers/{game_id}")
def update_pitchers(
    game_id: int,
    service: LineupService = Depends(get_lineup_service),
):
    """
    Pull the latest probable pitcher IDs for a game from the MLB Stats API
    and update the game record.
    """
    game = service.update_pitchers_for_game(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return {
        "game_id": game_id,
        "home_probable_pitcher_id": game.home_probable_pitcher_id,
        "away_probable_pitcher_id": game.away_probable_pitcher_id,
    }
