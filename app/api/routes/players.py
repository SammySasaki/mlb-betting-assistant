from typing import List, Optional

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_at_bat_service
from app.services.at_bat_service import AtBatService

router = APIRouter(prefix="/players", tags=["Players"])

_VALID_SELECT_FIELDS = {"hits", "at_bats", "batting_average", "at_bats_detail", "events"}


@router.get("/{player_id}/matchup")
def get_matchup_stats(
    player_id: int,
    opponent_id: int = Query(description="MLB player ID of the opponent"),
    position: str = Query(description="'B' if player_id is the batter, 'P' if pitcher"),
    season: Optional[int] = Query(default=None, description="Season year; omit for all seasons"),
    limit: Optional[int] = Query(default=None, description="Max number of at-bats to return"),
    select_fields: List[str] = Query(
        default=["hits", "at_bats", "batting_average"],
        description=(
            "Fields to include in the response. "
            "Options: hits, at_bats, batting_average, at_bats_detail, events"
        ),
    ),
    aggregate: Optional[str] = Query(
        default=None,
        description="Aggregation mode: 'sum', 'count', or omit for none",
    ),
    service: AtBatService = Depends(get_at_bat_service),
):
    """
    Retrieve batter-vs-pitcher head-to-head stats via Statcast.

    Set `position='B'` when `player_id` is the batter and `opponent_id` is the pitcher,
    or `position='P'` for the reverse.
    """
    return service.get_matchup_stats(
        player_id=player_id,
        position=position,
        opponent_id=opponent_id,
        limit=limit,
        season=season,
        select_fields=select_fields,
        aggregate=aggregate,
    )
