from datetime import date
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_game_repo
from app.api.schemas import GameResponse
from app.implementations.sqlalchemy_game_repository import GameRepository

router = APIRouter(prefix="/games", tags=["Games"])


@router.get("/{game_id}", response_model=GameResponse)
def get_game(game_id: int, repo: GameRepository = Depends(get_game_repo)):
    """Get a single game by ID."""
    game = repo.get_by_id(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return game


@router.get("", response_model=List[GameResponse])
def get_games(
    date: Optional[date] = Query(None, description="Filter by game date (YYYY-MM-DD)"),
    season: Optional[int] = Query(None, description="Filter by season year"),
    team: Optional[str] = Query(None, description="Filter by team name (home or away)"),
    repo: GameRepository = Depends(get_game_repo),
):
    """
    List games filtered by date, season, or team.
    Provide at least one of `date` or `season`.
    """
    if date:
        if team:
            game = repo.get_game_by_date_by_team(date, team)
            return [game] if game else []
        return repo.get_games_by_date(date)

    if season:
        games = repo.get_games_by_season(season)
        if team:
            games = [
                g for g in games if g.home_team == team or g.away_team == team
            ]
        return games

    raise HTTPException(
        status_code=400,
        detail="Provide at least one of the query parameters: date, season",
    )
