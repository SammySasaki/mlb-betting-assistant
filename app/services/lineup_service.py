from typing import List
from sqlalchemy.orm import Session
from app.db_models import Lineups, PlayerGameStats
from app.interfaces.iapi_client import IApiClient
from app.interfaces.ilineup_repository import ILineupRepository
from app.interfaces.iplayer_game_stats_repository import IPlayerGameStatsRepository
from app.interfaces.igame_repository import IGameRepository
from infra.db.init_db import SessionLocal


class LineupService:
    def __init__(
        self,
        player_stats_repo: IPlayerGameStatsRepository,
        lineup_repo: ILineupRepository,
        mlb_client: IApiClient,
        game_repo: IGameRepository,
    ):
        self.player_stats_repo = player_stats_repo
        self.lineup_repo = lineup_repo
        self.mlb_client = mlb_client
        self.game_repo = game_repo

    def save_lineups_for_game(self, game_id: int):
        """Fetch lineup for a given game, enriched with season + last10 stats"""
        print(f"Fetching lineup for game_id {game_id}")
        data = self.mlb_client.get_boxscore(game_id)

        game = self.game_repo.get_by_id(game_id)
        if not data or not game:
            print(f"Warning: No data found for game_id {game_id}")
            return
        date = game.date
        entries: List[Lineups] = []
        for team_key in ["home", "away"]:
            team_data = data["teams"][team_key]
            season = team_data["team"]["season"]
            home_sp, away_sp = self.mlb_client.get_probable_pitchers(game_id)
            game.home_probable_pitcher_id = home_sp
            game.away_probable_pitcher_id = away_sp
            self.game_repo.update(game)
            for player_id in team_data.get("batters", []):
                player_obj = team_data["players"][f"ID{player_id}"]
                if player_obj.get("battingOrder"):  # Only starting hitters
                    batting_order = player_obj.get("battingOrder")
                    position = player_obj.get("position", {}).get("abbreviation")

                    # --- Season stats before game date ---
                    season_games: List[PlayerGameStats] = self.player_stats_repo.get_aggregate_stats(
                        player_id, season_year=season, before_date=date
                    ) or []

                    # --- Last 10 games before game date ---
                    last10_games: List[PlayerGameStats] = self.player_stats_repo.get_last_n_games_stats(
                        player_id, n=10, before_date=date
                    )

                    # --- Compute season totals ---
                    season_totals = {
                        "hits": sum((getattr(g, "hits", 0) or 0) for g in season_games),
                        "at_bats": sum((getattr(g, "at_bats", 0) or 0) for g in season_games),
                        "doubles": sum((getattr(g, "doubles", 0) or 0) for g in season_games),
                        "triples": sum((getattr(g, "triples", 0) or 0) for g in season_games),
                        "home_runs": sum((getattr(g, "home_runs", 0) or 0) for g in season_games),
                        "walks": sum((getattr(g, "walks_batting", 0) or 0) for g in season_games),
                        "rbis": sum((getattr(g, "rbis", 0) or 0) for g in season_games),
                    }

                    # --- Compute last 10 games totals for recent OPS ---
                    last10_totals = {
                        "hits": sum((getattr(g, "hits", 0) or 0) for g in last10_games),
                        "at_bats": sum((getattr(g, "at_bats", 0) or 0) for g in last10_games),
                        "doubles": sum((getattr(g, "doubles", 0) or 0) for g in last10_games),
                        "triples": sum((getattr(g, "triples", 0) or 0) for g in last10_games),
                        "home_runs": sum((getattr(g, "home_runs", 0) or 0) for g in last10_games),
                        "walks": sum((getattr(g, "walks_batting", 0) or 0) for g in last10_games),      
                    }

                    recent_ops = self._compute_ops(last10_totals)

                    # --- Season OPS ---
                    ops_season = self._compute_ops(season_totals)

                    entry = Lineups(
                        game_id=game_id,
                        player_id=player_id,
                        batting_order=int(batting_order) if batting_order else None,
                        defensive_position=position,
                        avg_season=self._safe_div(season_totals.get("hits", 0), season_totals.get("at_bats", 0)),
                        obp_season=self._compute_obp(season_totals),
                        slg_season=self._compute_slg(season_totals),
                        ops_season=ops_season,
                        home_runs=season_totals.get("home_runs", 0),
                        rbis=season_totals.get("rbis", 0),
                        recent_ops=recent_ops,
                    )
                    entries.append(entry)

        # Save lineup entries in DB
        self.lineup_repo.upsert_lineup_entries(entries)
    
    def save_lineups_by_date(self, game_date):
        """Fetch lineups for all games on a given date"""
        print(f"Fetching lineups for games on {game_date}")
        games = self.game_repo.get_games_by_date(game_date)
        for game in games:
            self.save_lineups_for_game(game.id)

    def update_pitchers_for_game(self, game_id: int):
        game = self.game_repo.get_by_id(game_id)
        home_pitcher_id, away_pitcher_id = self.mlb_client.get_probable_pitchers(game_id)
        game.home_probable_pitcher_id = home_pitcher_id
        game.away_probable_pitcher_id = away_pitcher_id
        self.game_repo.update(game)
        return game
        
    # --- Helpers ---
    def _safe_div(self, num, den):
        return (num / den) if den else None

    def _compute_obp(self, stats):
        ab = stats.get("at_bats", 0)
        bb = stats.get("walks", 0)
        hits = stats.get("hits", 0)
        return ((hits + bb) / (ab + bb)) if (ab + bb) > 0 else None

    def _compute_slg(self, stats):
        ab = stats.get("at_bats", 0)
        if ab == 0:
            return None
        hits = stats.get("hits", 0)
        doubles = stats.get("doubles", 0)
        triples = stats.get("triples", 0)
        hr = stats.get("home_runs", 0)
        return (
            hits - doubles - triples - hr
            + 2 * doubles + 3 * triples + 4 * hr
        ) / ab

    def _compute_ops(self, stats):
        obp = self._compute_obp(stats)
        slg = self._compute_slg(stats)
        return (obp + slg) if (obp is not None and slg is not None) else None
    
    if __name__ == "__main__":
        
        with SessionLocal() as session:
            from app.implementations.sqlalchemy_lineup_repository import SQLAlchemyLineupRepository
            from app.implementations.sqlalchemy_game_repository import GameRepository
            from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
            from app.implementations.mlb_api_client import StatsApiClient
            from app.services.lineup_service import LineupService

            player_stats_repo = SqlAlchemyPlayerGameStatsRepository(session)
            lineup_repo = SQLAlchemyLineupRepository(session)
            game_repo = GameRepository(session)
            mlb_client = StatsApiClient()

            service = LineupService(
                player_stats_repo=player_stats_repo,
                lineup_repo=lineup_repo,
                mlb_client=mlb_client,
                game_repo=game_repo,
            )

            from datetime import date, timedelta
            # service.save_lineups_by_date(date(2025, 9, 19))

            
            START = date(2021, 3, 7)
            END = date(2025, 9, 20)
            current = START
            while current <= END:
                print(f"Processing lineups for {current}")
                service.save_lineups_by_date(current)
                current += timedelta(days=1)