from datetime import date, timezone, datetime
from difflib import SequenceMatcher
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import Player, Game
from app.implementations.sqlalchemy_news_repository import SqlAlchemyNewsRepository
from app.interfaces.i_news_client import INewsClient
from app.interfaces.i_news_repository import INewsRepository

MLB_TEAMS = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox",
    "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians",
    "Colorado Rockies", "Detroit Tigers", "Houston Astros", "Kansas City Royals",
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins", "Milwaukee Brewers",
    "Minnesota Twins", "New York Mets", "New York Yankees", "Oakland Athletics",
    "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres",
    "San Francisco Giants", "Seattle Mariners", "St. Louis Cardinals",
    "Tampa Bay Rays", "Texas Rangers", "Toronto Blue Jays", "Washington Nationals",
]

# Short aliases that also appear in headlines
_TEAM_ALIASES: dict[str, str] = {
    "Diamondbacks": "Arizona Diamondbacks",
    "D-backs": "Arizona Diamondbacks",
    "Braves": "Atlanta Braves",
    "Orioles": "Baltimore Orioles",
    "Red Sox": "Boston Red Sox",
    "Cubs": "Chicago Cubs",
    "White Sox": "Chicago White Sox",
    "Reds": "Cincinnati Reds",
    "Guardians": "Cleveland Guardians",
    "Rockies": "Colorado Rockies",
    "Tigers": "Detroit Tigers",
    "Astros": "Houston Astros",
    "Royals": "Kansas City Royals",
    "Angels": "Los Angeles Angels",
    "Dodgers": "Los Angeles Dodgers",
    "Marlins": "Miami Marlins",
    "Brewers": "Milwaukee Brewers",
    "Twins": "Minnesota Twins",
    "Mets": "New York Mets",
    "Yankees": "New York Yankees",
    "Athletics": "Oakland Athletics",
    "Phillies": "Philadelphia Phillies",
    "Pirates": "Pittsburgh Pirates",
    "Padres": "San Diego Padres",
    "Giants": "San Francisco Giants",
    "Mariners": "Seattle Mariners",
    "Cardinals": "St. Louis Cardinals",
    "Rays": "Tampa Bay Rays",
    "Rangers": "Texas Rangers",
    "Blue Jays": "Toronto Blue Jays",
    "Nationals": "Washington Nationals",
}


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


class NewsService:

    def __init__(self, news_client: INewsClient, repo: INewsRepository, db: Session):
        self.news_client = news_client
        self.repo = repo
        self.db = db

    def scrape_and_store(self) -> dict:
        result = self.news_client.scrape()
        articles = result.get("articles", [])
        enriched = self._enrich(articles)
        inserted, skipped = self.repo.upsert_many(enriched)
        return {
            "inserted": inserted,
            "skipped_duplicates": skipped,
            "sources_scraped": result.get("sources_scraped", 0),
            "total_from_go": result.get("total_articles", 0),
            "scrape_duration_ms": result.get("scrape_duration_ms", 0),
        }

    def get_recent(self, limit: int = 20, category: Optional[str] = None):
        return self.repo.get_recent(limit=limit, category=category)

    def get_by_game_id(self, game_id: int):
        return self.repo.get_by_game_id(game_id)

    # ------------------------------------------------------------------

    def _enrich(self, articles: list[dict]) -> list[dict]:
        players = self.db.query(Player).all()
        today = datetime.now(timezone.utc).date()
        todays_games = (
            self.db.query(Game)
            .filter(Game.date == today)
            .all()
        )

        enriched = []
        for article in articles:
            text = f"{article.get('headline', '')} {article.get('body', '')}"
            article["teams"] = self._find_teams(text)
            article["player_id"] = self._find_player_id(text, players, article.get("player_name", ""))
            article["game_id"] = self._find_game_id(article["teams"], todays_games)
            enriched.append(article)
        return enriched

    def _find_teams(self, text: str) -> list[str]:
        found = []
        for team in MLB_TEAMS:
            if team.lower() in text.lower():
                found.append(team)
                continue
        for alias, canonical in _TEAM_ALIASES.items():
            if alias.lower() in text.lower() and canonical not in found:
                found.append(canonical)
        return found

    def _find_player_id(self, text: str, players: list[Player], player_name: str = "") -> Optional[int]:
        # Rotowire supplies the player name explicitly — use it for an exact match first
        if player_name:
            name_lower = player_name.lower()
            for player in players:
                if player.name.lower() == name_lower:
                    return player.id
            # Fall back to last-name match within the explicit name
            last = player_name.split()[-1].lower()
            for player in players:
                if player.name.lower().endswith(last):
                    return player.id

        # General text scan for other sources
        text_lower = text.lower()
        for player in players:
            if player.name.lower() in text_lower:
                return player.id
        for player in players:
            last_name = player.name.split()[-1]
            if len(last_name) >= 4 and last_name.lower() in text_lower:
                return player.id
        return None

    def _find_game_id(self, teams: list[str], games: list[Game]) -> Optional[int]:
        if not teams:
            return None
        for game in games:
            if game.home_team in teams or game.away_team in teams:
                return game.id
        return None
