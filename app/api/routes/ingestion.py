from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_ingestion_service
from app.api.schemas import IngestDayRequest, IngestRangeRequest
from app.services.ingestion_service import MLBIngestionService

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


@router.post("/day")
def ingest_day(
    body: IngestDayRequest,
    service: MLBIngestionService = Depends(get_ingestion_service),
):
    """
    Ingest all MLB data for a single calendar day.

    Fetches games, box-score stats, lineups, probable pitchers,
    team/pitcher/bullpen features, and weather for the given date.
    All writes are idempotent (upserts).
    """
    return service.ingest_day(body.date)


@router.post("/range")
def ingest_range(
    body: IngestRangeRequest,
    service: MLBIngestionService = Depends(get_ingestion_service),
):
    """
    Ingest all MLB data for every day in [start_date, end_date] inclusive.
    """
    return service.ingest_dates(body.start_date, body.end_date)


@router.post("/upcoming")
def ingest_upcoming(
    body: IngestDayRequest,
    service: MLBIngestionService = Depends(get_ingestion_service),
):
    """
    Store game records for upcoming or in-progress games (no scores required yet).

    Call this before games are played so the odds and prediction pipeline
    has game rows to work with. After games finish, call POST /ingestion/day
    to fill in scores, box-score stats, and computed features.
    """
    return service.ingest_upcoming_day(body.date)


@router.post("/backfill-weather")
def backfill_weather(
    batch_size: int = Query(default=10, ge=1, description="Games to commit per transaction"),
    service: MLBIngestionService = Depends(get_ingestion_service),
):
    """
    Fetch weather for every historical game that is missing a weather record.

    Processes games oldest-first. Stops gracefully if the weather API credit
    limit is exhausted, committing progress made so far.
    """
    return service.backfill_missing_weather(batch_size=batch_size)
