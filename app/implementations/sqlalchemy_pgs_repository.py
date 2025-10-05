from typing import Optional, List, Tuple
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db_models import PlayerGameStats, Game
from app.interfaces.iplayer_game_stats_repository import IPlayerGameStatsRepository


class SqlAlchemyPlayerGameStatsRepository(IPlayerGameStatsRepository):
    """SQLAlchemy implementation of PlayerGameStats repository."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_player_and_season(self, player_id: int, season_year: int) -> List[PlayerGameStats]:
        return (
            self.session.query(PlayerGameStats)
            .join(Game, Game.id == PlayerGameStats.game_id)
            .filter(PlayerGameStats.player_id == player_id, Game.season_year == season_year)
            .all()
        )
    
    def get_batters_by_team_and_season_and_date(self, team: str, season_year: int, before_date: date) -> List[PlayerGameStats]:
        return (
            self.session.query(PlayerGameStats)
            .join(Game)
            .filter(
                PlayerGameStats.team == team,
                Game.date < before_date,
                Game.season_year == season_year,
                PlayerGameStats.at_bats != None
            )
            .all()
        )

    def get_aggregate_stats(
        self, player_id: int, season_year: Optional[int] = None, before_date: Optional[date] = None
    ) -> Optional[dict]:
        """
        Returns all PlayerGameStats for a player in a season (optionally before a date).
        """
        query = (
            self.session.query(PlayerGameStats)
            .join(Game, Game.id == PlayerGameStats.game_id)
            .filter(PlayerGameStats.player_id == player_id)
        )
        if season_year is not None:
            query = query.filter(Game.season_year == season_year)
        if before_date:
            query = query.filter(Game.date < before_date)
        return query.all()

    def get_last_n_games_stats(
        self, player_id: int, n: int, before_date: Optional[date] = None
    ) -> Optional[dict]:
        """
        Returns PlayerGameStats objects for the last n games for a player (optionally before a date).
        """
        query = (
            self.session.query(PlayerGameStats)
            .join(Game, Game.id == PlayerGameStats.game_id)
            .filter(PlayerGameStats.player_id == player_id)
            .order_by(Game.date.desc())
        )
        if before_date:
            query = query.filter(Game.date < before_date)
        return query.limit(n).all()