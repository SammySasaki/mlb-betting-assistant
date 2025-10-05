from typing import Optional
from sqlalchemy.orm import Session
from app.db_models import PitcherFeatures, BullpenFeatures, TeamFeatures
from app.interfaces.ifeatures_repository import IFeaturesRepository

class SqlAlchemyFeaturesRepository(IFeaturesRepository):
    def __init__(self, session: Session):
        self.session = session

    def get_by_game_and_player(self, game_id: int, player_id: int) -> Optional[PitcherFeatures]:
        return (
            self.session.query(PitcherFeatures)
            .filter_by(game_id=game_id, player_id=player_id)
            .first()
        )
    
    def get_team_features_by_game_and_team(self, game_id: int, team: str) -> Optional[TeamFeatures]:
        return (
            self.session.query(TeamFeatures)
            .filter_by(game_id=game_id, team=team)
            .first()
        )
    
    def get_bullpen_features_by_game_and_team(self, game_id: int, team: str) -> Optional[BullpenFeatures]:
        """Fetch pitcher features for a given game and player (probable SP)."""
        return (
            self.session.query(BullpenFeatures)
            .filter_by(game_id=game_id, team=team)
            .first()
        )