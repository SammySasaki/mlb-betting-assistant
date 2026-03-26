from typing import Optional
from sqlalchemy.orm import Session
from app.interfaces.iplayer_repository import IPlayerRepository
from app.db.models import Player

class SQLAlchemyPlayerRepository(IPlayerRepository):
    def __init__(self, session: Session):
        self.session = session

    def get_player_name(self, player_id: int) -> Optional[str]:
        player = self.session.query(Player).filter(Player.id == player_id).one_or_none()
        if player:
            return player.name
        return None

    def get_by_id(self, player_id: int) -> Optional[Player]:
        return self.session.query(Player).filter(Player.id == player_id).one_or_none()

    def flush(self) -> None:
        self.session.commit()

    def upsert(self, player_id: int, name: str, team: str, position: str, throwing_hand: Optional[str] = None, existing: Optional[Player] = None) -> None:
        player = existing if existing is not None else self.session.query(Player).filter(Player.id == player_id).one_or_none()
        if player:
            player.team = team
            player.position = position
        else:
            self.session.add(Player(id=player_id, name=name, team=team, position=position, throwing_hand=throwing_hand))