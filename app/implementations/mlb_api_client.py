import requests
from datetime import date
from typing import Dict, Any
from app.interfaces.iapi_client import IApiClient


class StatsApiClient(IApiClient):
    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self._player_arm_cache: dict[int, str] = {}

    def get_schedule(self, target_date: date) -> list:
        """Return list of game dicts for a given date."""
        resp = self.session.get(f"{self.BASE_URL}/schedule", params={
            "sportId": 1,
            "date": target_date.strftime("%Y-%m-%d"),
            "hydrate": "team,linescore",
        })
        resp.raise_for_status()
        data = resp.json()
        return data.get("dates", [])[0].get("games", []) if data.get("dates") else []

    def get_boxscore(self, game_id: int) -> Dict[str, Any]:
        """Fetch boxscore for a single game."""
        resp = self.session.get(f"{self.BASE_URL}/game/{game_id}/boxscore")
        resp.raise_for_status()
        return resp.json()

    def get_player(self, player_id: int) -> dict | None:
        """Return the people[0] dict for a player, or None if not found."""
        resp = self.session.get(f"{self.BASE_URL}/people/{player_id}/")
        if resp.status_code != 200:
            return None
        try:
            return resp.json()["people"][0]
        except (KeyError, IndexError):
            return None

    def get_player_arm(self, player_id: int) -> str | None:
        """Return pitching-hand code ('L'/'R'/'S') for a player, or None."""
        if player_id in self._player_arm_cache:
            return self._player_arm_cache[player_id]
        person = self.get_player(player_id)
        if not person:
            return None
        try:
            code = person["pitchHand"]["code"]
            self._player_arm_cache[player_id] = code
            return code
        except (KeyError, IndexError):
            return None

    def get_league_leaders(self, stat_category: str, season: int, stat_group: str = "pitching", limit: int = 10) -> list[dict]:
        """Return ranked league leaders for a given stat category."""
        resp = self.session.get(f"{self.BASE_URL}/stats/leaders", params={
            "leaderCategories": stat_category,
            "season": season,
            "sportId": 1,
            "limit": limit,
            "statGroup": stat_group,
        })
        resp.raise_for_status()
        leaders = resp.json().get("leagueLeaders", [{}])[0].get("leaders", [])
        return [
            {"rank": l["rank"], "name": l["person"]["fullName"], "value": l["value"]}
            for l in leaders
        ]

    def get_player_season_stats(self, player_id: int, season: int, stat_group: str = "hitting") -> dict:
        """Return season stat split for a single player."""
        resp = self.session.get(f"{self.BASE_URL}/people/{player_id}/stats", params={
            "stats": "season",
            "season": season,
            "group": stat_group,
        })
        resp.raise_for_status()
        splits = resp.json().get("stats", [{}])[0].get("splits", [])
        return splits[0].get("stat", {}) if splits else {}

    def get_probable_pitchers(self, game_id: int) -> tuple[int | None, int | None]:
        """Return (home_pitcher_id, away_pitcher_id) for a game."""
        try:
            data = self.session.get(f"{self.BASE_URL}/schedule", params={
                "sportId": 1,
                "gamePk": game_id,
                "hydrate": "probablePitcher",
            }).json()
            teams = data["dates"][0]["games"][0]["teams"]
            home_id = teams["home"].get("probablePitcher", {}).get("id")
            away_id = teams["away"].get("probablePitcher", {}).get("id")
            return home_id, away_id
        except (KeyError, IndexError):
            return None, None