from fastapi import APIRouter, Depends

from app.api.dependencies import get_odds_service
from app.api.schemas import OddsDateRequest
from app.services.odds_service import OddsService

router = APIRouter(prefix="/odds", tags=["Odds"])


@router.post("/lines")
def ingest_lines(
    body: OddsDateRequest,
    service: OddsService = Depends(get_odds_service),
):
    """
    Fetch and persist current totals + moneyline odds for games on the given date.

    Intended for upcoming games (today or tomorrow). Fuzzy-matches team names
    from the odds provider to Game rows in the database.
    """
    return service.ingest_lines_for_date(body.date)


@router.post("/historical")
def ingest_historical_lines(
    body: OddsDateRequest,
    service: OddsService = Depends(get_odds_service),
):
    """
    Fetch and persist historical totals lines for a past date.

    Only totals (over/under) lines are available historically;
    moneyline prices are skipped.
    """
    return service.ingest_historical_lines_for_date(body.date)
