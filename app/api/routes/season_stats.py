from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.dependencies import get_season_stats_repo, get_season_stats_service
from app.api.schemas import PlayerSeasonStatsResponse
from app.implementations.sqlalchemy_player_season_stats_repository import SQLAlchemyPlayerSeasonStatsRepository
from app.services.season_stats_service import SeasonStatsService

router = APIRouter(prefix="/season-stats", tags=["Season Stats"])


@router.get("", response_model=List[PlayerSeasonStatsResponse])
def get_season_stats(
    season: int = Query(...),
    player_id: Optional[int] = Query(None),
    team: Optional[str] = Query(None),
    repo: SQLAlchemyPlayerSeasonStatsRepository = Depends(get_season_stats_repo),
):
    """Return season stats for all players, or filtered by player_id or team."""
    if player_id is not None:
        row = repo.get_by_player_and_season(player_id, season)
        return [row] if row else []

    if team is not None:
        return repo.get_by_team_and_season(team, season)

    return repo.get_by_season(season)


@router.post("/calculate")
def calculate_season_stats(
    season: int = Query(...),
    service: SeasonStatsService = Depends(get_season_stats_service),
):
    """Trigger a full recalculation of season stats for all players in the given season.

    Use this after a backfill or if the table gets out of sync.
    """
    return service.calculate_season_stats(season)
