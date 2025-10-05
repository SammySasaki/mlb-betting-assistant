from typing import List, Optional
from datetime import date
from sqlalchemy.orm import Session
from sqlalchemy import func, case
from app.db_models import Game
from app.interfaces.igame_repository import IGameRepository


class GameRepository(IGameRepository):
    """SQLAlchemy implementation of IGameRepository."""

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, game_id: int) -> Optional[Game]:
        return self.session.query(Game).filter(Game.id == game_id).first()
    

    def get_games_by_date(self, game_date: date) -> List[Game]:
        return (
            self.session.query(Game)
            .filter(Game.date == game_date)
            .order_by(Game.start_hour_utc.asc())
            .all()
        )

    def get_games_by_season(self, season_year: int) -> List[Game]:
        return (
            self.session.query(Game)
            .filter(Game.season_year == season_year)
            .order_by(Game.date.desc())
            .all()
        )

    def add(self, game: Game) -> Game:
        self.session.add(game)
        self.session.commit()
        self.session.refresh(game)
        return game

    def update(self, game: Game) -> Game:
        self.session.merge(game)
        self.session.commit()
        return game

    def delete(self, game_id: int) -> None:
        self.session.query(Game).filter(Game.id == game_id).delete()
        self.session.commit()

    def get_game_by_date_by_team(self, game_date: date, team: str) -> Optional[Game]:
        return (
            self.session.query(Game)
            .filter(
                Game.date == game_date,
                ((Game.home_team == team) | (Game.away_team == team))
            )
            .first()
        )
        

    def get_games_by_team_before_date(
        self, team: str, before_date: date, season_year: int
    ) -> List[Game]:
        return (
            self.session.query(Game)
            .filter(
                ((Game.home_team == team) | (Game.away_team == team)),
                Game.date < before_date,
                Game.season_year == season_year,
            )
            .all()
        )
    
    def get_team_record_before_date(self, team_name: str, date, season_year: int):
        result = self.session.query(
            func.sum(
                case(
                    (
                        ( (Game.home_team == team_name) & (Game.home_score > Game.away_score) ) |
                        ( (Game.away_team == team_name) & (Game.away_score > Game.home_score) ),
                        1
                    ),
                    else_=0
                )
            ).label("wins"),
            func.sum(
                case(
                    (
                        ( (Game.home_team == team_name) & (Game.home_score < Game.away_score) ) |
                        ( (Game.away_team == team_name) & (Game.away_score < Game.home_score) ),
                        1
                    ),
                    else_=0
                )
            ).label("losses"),
        ).filter(
            Game.date < date,
            Game.season_year == season_year
        ).one()

        return (result.wins or 0, result.losses or 0)