from collections import defaultdict
from datetime import datetime, date

import pytz
from sqlalchemy.orm import Session

from app.graph.state import GraphState
from app.implementations.sqlalchemy_news_repository import SqlAlchemyNewsRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.sqlalchemy_game_repository import GameRepository


class ContextAgent:
    def __init__(self, db_session: Session, n_recent_games: int = 5):
        self.news_repo = SqlAlchemyNewsRepository(db_session)
        self.pgs_repo = SqlAlchemyPlayerGameStatsRepository(db_session)
        self.game_repo = GameRepository(db_session)
        self.n = n_recent_games

    def gather_context(self, state: GraphState) -> GraphState:
        preds = state.get("predictions_data") or []
        eastern = pytz.timezone("America/New_York")
        today = datetime.now(eastern).date()
        season_year = today.year

        context_data = {}
        for rec in preds:
            game_id = rec["game_id"]
            home_team = rec["home_team"]
            away_team = rec["away_team"]

            home_stats = self._recent_team_batting(home_team, today, season_year)
            away_stats = self._recent_team_batting(away_team, today, season_year)

            news_items = self.news_repo.get_recent_by_teams([home_team, away_team], hours=48)
            news = [
                f"[{item.category or 'NEWS'}] {item.headline}"
                for item in news_items[:10]
            ]

            context_data[str(game_id)] = {
                "home_team_stats": home_stats,
                "away_team_stats": away_stats,
                "news": news,
            }

        return {**state, "context_data": context_data}

    def _recent_team_batting(self, team: str, before_date: date, season_year: int) -> list:
        rows = self.pgs_repo.get_batters_by_team_and_season_and_date(
            team=team, season_year=season_year, before_date=before_date
        )
        if not rows:
            return []

        by_game = defaultdict(list)
        for r in rows:
            by_game[r.game_id].append(r)

        # Sort game_ids by game date descending, take most recent N
        def game_date(gid):
            g = self.game_repo.get_by_id(gid)
            return g.date if g else date.min

        recent_ids = sorted(by_game.keys(), key=game_date, reverse=True)[: self.n]

        totals = {"hits": 0, "at_bats": 0, "home_runs": 0, "rbis": 0, "runs": 0}
        for gid in recent_ids:
            for r in by_game[gid]:
                totals["hits"] += r.hits or 0
                totals["at_bats"] += r.at_bats or 0
                totals["home_runs"] += r.home_runs or 0
                totals["rbis"] += r.rbis or 0
                totals["runs"] += r.runs or 0

        avg = round(totals["hits"] / totals["at_bats"], 3) if totals["at_bats"] else 0.0
        return [{"team": team, "last_n_games": len(recent_ids), **totals, "batting_avg": avg}]
