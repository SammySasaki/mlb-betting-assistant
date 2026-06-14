from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import NewsItem
from app.interfaces.i_news_repository import INewsRepository


class SqlAlchemyNewsRepository(INewsRepository):

    def __init__(self, session: Session):
        self.session = session

    def upsert_many(self, articles: list[dict]) -> tuple[int, int]:
        if not articles:
            return 0, 0

        inserted = 0
        skipped = 0
        for article in articles:
            stmt = (
                insert(NewsItem)
                .values(
                    source=article["source"],
                    headline=article["headline"],
                    body=article.get("body"),
                    url=article["url"],
                    published_at=article.get("published_at") or None,
                    category=article.get("category"),
                    teams=article.get("teams") or [],
                    player_id=article.get("player_id"),
                    game_id=article.get("game_id"),
                )
                .on_conflict_do_nothing(index_elements=["url"])
            )
            result = self.session.execute(stmt)
            if result.rowcount:
                inserted += 1
            else:
                skipped += 1

        self.session.commit()
        return inserted, skipped

    def get_recent(self, limit: int = 20, category: Optional[str] = None) -> list[NewsItem]:
        q = self.session.query(NewsItem)
        if category:
            q = q.filter(NewsItem.category == category)
        from sqlalchemy import func as sqlfunc
        sort_col = sqlfunc.coalesce(NewsItem.published_at, NewsItem.scraped_at)
        return q.order_by(sort_col.desc()).limit(limit).all()

    def get_recent_by_teams(self, teams: list[str], hours: int = 24) -> list[NewsItem]:
        if not teams:
            return []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            self.session.query(NewsItem)
            .filter(
                NewsItem.teams.overlap(teams),
                NewsItem.scraped_at >= cutoff,
            )
            .order_by(NewsItem.published_at.desc())
            .all()
        )

    def get_by_game_id(self, game_id: int) -> list[NewsItem]:
        return (
            self.session.query(NewsItem)
            .filter(NewsItem.game_id == game_id)
            .order_by(NewsItem.published_at.desc())
            .all()
        )
