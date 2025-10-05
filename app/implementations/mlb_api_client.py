import requests
from typing import Dict, Any
from app.db_models import Lineups
from app.interfaces.iapi_client import IApiClient
from app.interfaces.ilineup_repository import ILineupRepository
from app.interfaces.iplayer_game_stats_repository import IPlayerGameStatsRepository
from app.interfaces.iapi_client import IApiClient


class StatsApiClient(IApiClient):
    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()

    def get_boxscore(self, game_id: int) -> Dict[str, Any]:
        """
        Fetch the boxscore for a given MLB game.
        https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore
        """
        url = f"{self.BASE_URL}/game/{game_id}/boxscore"
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.json()
    
    def get_probable_pitchers(self, game_id: str):
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePk={game_id}&hydrate=probablePitcher"
        try:
            response = requests.get(url).json()
            home_pitcher = response["dates"][0]["games"][0]["teams"]["home"].get("probablePitcher", {}).get("id", None)
            away_pitcher = response["dates"][0]["games"][0]["teams"]["away"].get("probablePitcher", {}).get("id", None)
            if game_id == 776249:
                print(home_pitcher, away_pitcher)
            return (home_pitcher, away_pitcher)
        except (KeyError, IndexError, ValueError):
            return ("", "")