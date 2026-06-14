import logging
import traceback
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_news_service
from app.services.news_service import NewsService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/news", tags=["News"])


@router.post("/scrape")
def scrape_news(service: NewsService = Depends(get_news_service)):
    try:
        return service.scrape_and_store()
    except Exception as exc:
        logger.error("News scrape failed: %s\n%s", exc, traceback.format_exc())
        raise HTTPException(status_code=502, detail=f"Go news service error: {exc}")


@router.get("")
def get_recent_news(
    limit: int = 20,
    category: Optional[str] = None,
    service: NewsService = Depends(get_news_service),
):
    items = service.get_recent(limit=limit, category=category)
    return [_serialize(item) for item in items]


@router.get("/game/{game_id}")
def get_news_for_game(game_id: int, service: NewsService = Depends(get_news_service)):
    items = service.get_by_game_id(game_id)
    return [_serialize(item) for item in items]


def _serialize(item) -> dict:
    return {
        "id": item.id,
        "source": item.source,
        "headline": item.headline,
        "body": item.body,
        "url": item.url,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "scraped_at": item.scraped_at.isoformat() if item.scraped_at else None,
        "category": item.category,
        "teams": item.teams or [],
        "player_id": item.player_id,
        "game_id": item.game_id,
    }
