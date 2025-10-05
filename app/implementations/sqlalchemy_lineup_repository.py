from typing import List
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.db_models import Lineups, Game
from app.interfaces.ilineup_repository import ILineupRepository


class SQLAlchemyLineupRepository(ILineupRepository):
    """SQLAlchemy implementation of ILineupRepository"""

    def __init__(self, session: Session):
        self.session = session

    def get_lineup_by_game(self, game_id: int) -> List[Lineups]:
        return (
            self.session.query(Lineups)
            .filter(Lineups.game_id == game_id)
            .order_by(Lineups.batting_order.asc())
            .all()
        )

    def upsert_lineup_entries(self, entries: List[Lineups]) -> None:
        for entry in entries:
            existing = (
                self.session.query(Lineups)
                .filter(
                    Lineups.game_id == entry.game_id,
                    Lineups.batting_order == entry.batting_order
                )
                .one_or_none()
            )
            if existing:
                # Replace all relevant fields
                existing.player_id = entry.player_id
                existing.defensive_position = entry.defensive_position
                existing.avg_season = entry.avg_season
                existing.obp_season = entry.obp_season
                existing.slg_season = entry.slg_season
                existing.ops_season = entry.ops_season
                existing.home_runs = entry.home_runs
                existing.rbis = entry.rbis
                existing.recent_ops = entry.recent_ops
            else:
                self.session.add(entry)
        self.session.commit()

    def delete_lineup_for_game(self, game_id: int) -> None:
        self.session.query(Lineups).filter(Lineups.game_id == game_id).delete()
        self.session.commit()

    def get_lineups_by_date(self, game_date: date) -> List[Lineups]:
        return (
            self.session.query(Lineups)
            .join(Game, Lineups.game_id == Game.id)
            .filter(Game.date == game_date)
            .order_by(Game.id.asc(), Lineups.batting_order.asc())
            .all()
        )
    
    def get_lineups_by_date_by_team(self, game_date: date, team: str) -> List[Lineups]:
        """
        Get all lineup entries for a specific team on a given date.
        """
        return (
            self.session.query(Lineups)
            .join(Game, Lineups.game_id == Game.id)
            .filter(
                Game.date == game_date,
                or_(Game.home_team == team, Game.away_team == team)
            )
            .order_by(Game.id.asc(), Lineups.batting_order.asc())
            .all()
        )
