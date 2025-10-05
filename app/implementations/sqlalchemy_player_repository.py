from typing import Optional
from sqlalchemy.orm import Session
from app.interfaces.iplayer_repository import IPlayerRepository
from app.db_models import Player

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