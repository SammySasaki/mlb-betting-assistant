from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models import Player, PlayerSeasonStats
from app.interfaces.iplayer_season_stats_repository import IPlayerSeasonStatsRepository


class SQLAlchemyPlayerSeasonStatsRepository(IPlayerSeasonStatsRepository):

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_player_and_season(self, player_id: int, season_year: int) -> Optional[PlayerSeasonStats]:
        return (
            self.session.query(PlayerSeasonStats)
            .filter_by(player_id=player_id, season_year=season_year)
            .first()
        )

    def get_by_season(self, season_year: int) -> List[PlayerSeasonStats]:
        return (
            self.session.query(PlayerSeasonStats)
            .filter_by(season_year=season_year)
            .all()
        )

    def get_by_team_and_season(self, team: str, season_year: int) -> List[PlayerSeasonStats]:
        return (
            self.session.query(PlayerSeasonStats)
            .join(Player, Player.id == PlayerSeasonStats.player_id)
            .filter(Player.team == team, PlayerSeasonStats.season_year == season_year)
            .all()
        )

    def upsert(self, stats: PlayerSeasonStats) -> None:
        self.session.merge(stats)
        self.session.commit()
