"""
app/services/ingestion_service.py

Service layer for MLB data ingestion.

Public API
----------
    service = MLBIngestionService()                         # uses StatsApiClient by default
    service = MLBIngestionService(mlb_client=mock_client)  # injectable for testing

    service.ingest_day(date(2025, 9, 20))
    service.ingest_dates(date(2025, 9, 1), date(2025, 9, 20))

All writes are upserts – calling ingest_day on an already-ingested date
refreshes it with the latest API data.
"""

import logging
from datetime import date, timedelta, datetime
from sqlalchemy.orm import Session
from sqlalchemy import text, exists

from infra.db.init_db import SessionLocal
from app.db.models import (
    Game, Player, PlayerGameStats,
    TeamFeatures, PitcherFeatures, BullpenFeatures, Weather,
)
from app.interfaces.iapi_client import IApiClient
from app.implementations.mlb_api_client import StatsApiClient
from app.interfaces.iweather_client import IWeatherClient, WeatherApiLimitError
from app.implementations.visual_crossing_client import VisualCrossingWeatherClient
from app.services.lineup_service import LineupService
from app.implementations.sqlalchemy_lineup_repository import SQLAlchemyLineupRepository
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.sqlalchemy_player_repository import SQLAlchemyPlayerRepository
from app.utils.utils import venue_utc_offsets
from app.implementations.sqlalchemy_player_season_stats_repository import SQLAlchemyPlayerSeasonStatsRepository
from app.services.season_stats_service import SeasonStatsService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VENUE_TO_CITY: dict[str, str] = {
    "Fenway Park": "Boston, MA",
    "Yankee Stadium": "New York, NY",
    "Dodger Stadium": "Los Angeles, CA",
    "Wrigley Field": "Chicago, IL",
    "Truist Park": "Atlanta, GA",
    "Oriole Park at Camden Yards": "Baltimore, MD",
    "George M. Steinbrenner Field": "Tampa, FL",
    "Rogers Centre": "Toronto, ON",
    "Comerica Park": "Detroit, MI",
    "Kauffman Stadium": "Kansas City, MO",
    "Angel Stadium": "Anaheim, CA",
    "Daikin Park": "Houston, TX",
    "Target Field": "Minneapolis, MN",
    "Sutter Health Park": "Oakland, CA",
    "T-Mobile Park": "Seattle, WA",
    "Progressive Field": "Cleveland, OH",
    "Globe Life Field": "Arlington, TX",
    "Rate Field": "Chicago, IL",
    "PNC Park": "Pittsburgh, PA",
    "Citizens Bank Park": "Philadelphia, PA",
    "Petco Park": "San Diego, CA",
    "Oracle Park": "San Francisco, CA",
    "American Family Field": "Milwaukee, WI",
    "Busch Stadium": "St. Louis, MO",
    "Nationals Park": "Washington, DC",
    "Great American Ball Park": "Cincinnati, OH",
    "loanDepot park": "Miami, FL",
    "Chase Field": "Phoenix, AZ",
    "Coors Field": "Denver, CO",
    "Citi Field": "New York, NY",
}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ip_to_outs(ip: float) -> int:
    """Convert MLB innings-pitched notation (X.1, X.2) to total outs."""
    if ip is None:
        return 0
    try:
        ip_float = float(ip)
    except (ValueError, TypeError):
        return 0
    innings = int(ip_float)
    fraction = round((ip_float - innings) * 10)
    if fraction not in (0, 1, 2):
        raise ValueError(f"Invalid innings pitched format: {ip}")
    return innings * 3 + fraction


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MLBIngestionService:
    """Encapsulates the full MLB data ingestion pipeline."""

    def __init__(
        self,
        mlb_client: IApiClient | None = None,
        weather_client: IWeatherClient | None = None,
    ) -> None:
        self._mlb_client = mlb_client or StatsApiClient()
        self._weather_client = weather_client or VisualCrossingWeatherClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_day(self, target_date: date) -> dict:
        """
        Ingest all MLB data for a single calendar day.

        Steps (in order):
          1. Fetch regular-season games from the MLB Stats API.
          2. Upsert game records and player box-score stats.
          3. Save / refresh lineup records.
          4. Fill missing probable-pitcher IDs.
          5. Compute and upsert team / pitcher / bullpen features.
          6. Fetch weather for any game that doesn't yet have it.

        If data for this date already exists every step is idempotent –
        scores, stats and features are refreshed; weather is left as-is
        (historical data does not change).

        Returns
        -------
        dict with keys:
            date             – ISO date string
            games_processed  – number of regular-season games stored
            errors           – list of error message strings
        """
        logger.info("Ingestion starting for %s", target_date)
        summary: dict = {
            "date": str(target_date),
            "games_processed": 0,
            "errors": [],
        }

        with SessionLocal() as session:
            # ── Step 1 & 2: games + player stats ──────────────────────
            raw_games = self._mlb_client.get_schedule(target_date)
            stored_games: list[Game] = []

            for g in raw_games:
                if g.get("gameType") != "R":
                    continue
                try:
                    game = self._store_game(session, g)
                    if game:
                        boxscore = self._mlb_client.get_boxscore(g["gamePk"])
                        self._store_player_stats(session, game.id, boxscore)
                        stored_games.append(game)
                except Exception as exc:
                    session.rollback()
                    msg = f"Game {g.get('gamePk')}: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)

            # ── Step 3: lineups ────────────────────────────────────────
            try:
                lineup_svc = self._make_lineup_service(session)
                lineup_svc.save_lineups_by_date(target_date)
            except Exception as exc:
                session.rollback()
                msg = f"Lineups: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

            session.commit()
            summary["games_processed"] = len(stored_games)

            # Reload games from DB so downstream steps see committed rows
            games_in_db = session.query(Game).filter(Game.date == target_date).all()

            # ── Step 4: probable pitchers ──────────────────────────────
            try:
                self._ingest_sps_for_games(session, games_in_db)
            except Exception as exc:
                msg = f"Starting pitchers: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

            # ── Step 5: features ───────────────────────────────────────
            try:
                games_in_db = session.query(Game).filter(Game.date == target_date).all()
                self._update_features_for_games(session, games_in_db, target_date)
            except Exception as exc:
                msg = f"Features: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

            # ── Step 6: weather ────────────────────────────────────────
            try:
                games_in_db = session.query(Game).filter(Game.date == target_date).all()
                self._ingest_weather_for_games(session, games_in_db)
            except Exception as exc:
                msg = f"Weather: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

            # ── Step 7: season stats ───────────────────────────────────
            try:
                season_stats_repo = SQLAlchemyPlayerSeasonStatsRepository(session)
                SeasonStatsService(session, season_stats_repo).update_for_date(target_date)
            except Exception as exc:
                msg = f"Season stats: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

        logger.info(
            "Ingestion complete for %s – %d games, %d error(s)",
            target_date, summary["games_processed"], len(summary["errors"]),
        )
        return summary

    def ingest_dates(self, start_date: date, end_date: date) -> dict:
        """
        Ingest data for every day in [start_date, end_date] inclusive.

        Returns
        -------
        dict with keys:
            days          – list of per-day summary dicts from ingest_day()
            total_games   – aggregate game count
            total_errors  – aggregate error count
        """
        aggregate: dict = {"days": [], "total_games": 0, "total_errors": 0}
        current = start_date
        while current <= end_date:
            day_summary = self.ingest_day(current)
            aggregate["days"].append(day_summary)
            aggregate["total_games"] += day_summary["games_processed"]
            aggregate["total_errors"] += len(day_summary["errors"])
            current += timedelta(days=1)
        return aggregate

    def ingest_upcoming_day(self, target_date: date) -> dict:
        """Store game records for upcoming or in-progress games that have no scores yet.

        Unlike ingest_day(), this method:
          - Stores game records even when scores are absent.
          - Resolves probable pitcher IDs and ensures those Player rows exist
            so FK constraints are satisfied before features are computed.
          - Saves lineup data for the day.

        Call this before games are played so the rest of the pipeline
        (odds ingestion, pre-game feature computation) has game rows to work
        with.  After games finish, call ingest_day() to fill in scores,
        box-score stats, and computed features.

        Returns
        -------
        dict with keys:
            date         – ISO date string
            games_stored – number of game records created or found
            errors       – list of error message strings
        """
        logger.info("Upcoming ingestion starting for %s", target_date)
        summary: dict = {"date": str(target_date), "games_stored": 0, "errors": []}

        with SessionLocal() as session:
            raw_games = self._mlb_client.get_schedule(target_date)

            for g in raw_games:
                if g.get("gameType") != "R":
                    continue
                try:
                    game = self._store_upcoming_game(session, g)
                    if game:
                        summary["games_stored"] += 1
                except Exception as exc:
                    session.rollback()
                    msg = f"Game {g.get('gamePk')}: {exc}"
                    logger.error(msg)
                    summary["errors"].append(msg)

            try:
                self._make_lineup_service(session).save_lineups_by_date(target_date)
            except Exception as exc:
                session.rollback()
                summary["errors"].append(f"Lineups: {exc}")

            try:
                games_in_db = session.query(Game).filter(Game.date == target_date).all()
                self._update_features_for_games(session, games_in_db, target_date)
            except Exception as exc:
                msg = f"Features: {exc}"
                logger.error(msg)
                summary["errors"].append(msg)

            session.commit()

        logger.info(
            "Upcoming ingestion complete for %s – %d games, %d error(s)",
            target_date, summary["games_stored"], len(summary["errors"]),
        )
        return summary

    # ------------------------------------------------------------------
    # Private – DB write helpers
    # ------------------------------------------------------------------

    def _make_lineup_service(self, session: Session) -> LineupService:
        return LineupService(
            player_stats_repo=SqlAlchemyPlayerGameStatsRepository(session),
            lineup_repo=SQLAlchemyLineupRepository(session),
            mlb_client=self._mlb_client,
            game_repo=GameRepository(session),
            player_repo=SQLAlchemyPlayerRepository(session),
        )

    def _store_upcoming_game(self, session: Session, game_data: dict) -> Game | None:
        """Create or refresh a game record even when scores are not yet available.

        Also resolves probable pitchers and ensures their Player rows exist so
        downstream FK constraints are satisfied.
        """
        game_id = game_data["gamePk"]
        game_date = game_data["officialDate"]
        home_team = game_data["teams"]["home"]["team"]["name"]
        away_team = game_data["teams"]["away"]["team"]["name"]
        dt = datetime.fromisoformat(game_data["gameDate"].replace("Z", "+00:00"))
        start_hour_utc = dt.hour
        season_year = datetime.strptime(game_date, "%Y-%m-%d").year

        home_pid, away_pid = self._mlb_client.get_probable_pitchers(game_id)

        if home_pid:
            self._ensure_player_in_db(session, home_pid, home_team)
        if away_pid:
            self._ensure_player_in_db(session, away_pid, away_team)

        existing = session.query(Game).filter_by(id=game_id).first()
        if not existing:
            game = Game(
                id=game_id,
                date=game_date,
                home_team=home_team,
                away_team=away_team,
                start_hour_utc=start_hour_utc,
                timestamp_utc=dt,
                home_probable_pitcher_id=home_pid,
                away_probable_pitcher_id=away_pid,
                venue=game_data.get("venue", {}).get("name"),
                season_year=season_year,
            )
            session.add(game)
            return game

        # Refresh pitcher IDs if they've now been announced
        if existing.home_probable_pitcher_id is None and home_pid:
            existing.home_probable_pitcher_id = home_pid
        if existing.away_probable_pitcher_id is None and away_pid:
            existing.away_probable_pitcher_id = away_pid
        existing.timestamp_utc = dt
        return existing

    def _ensure_player_in_db(self, session: Session, player_id: int, team: str) -> None:
        """Create a Player row for player_id if one doesn't already exist."""
        if session.query(Player).filter_by(id=player_id).first():
            return
        person = self._mlb_client.get_player(player_id)
        if not person:
            return
        session.add(Player(
            id=player_id,
            name=person.get("fullName"),
            team=team,
            position=person.get("primaryPosition", {}).get("abbreviation"),
            throwing_hand=person.get("pitchHand", {}).get("code"),
        ))

    def _store_game(self, session: Session, game_data: dict) -> Game | None:
        game_id = game_data["gamePk"]
        game_date = game_data["officialDate"]
        home_team = game_data["teams"]["home"]["team"]["name"]
        away_team = game_data["teams"]["away"]["team"]["name"]

        try:
            home_score = game_data["teams"]["home"]["score"]
            away_score = game_data["teams"]["away"]["score"]
        except KeyError:
            return None  # game hasn't been played yet

        total_runs = home_score + away_score
        dt = datetime.fromisoformat(game_data["gameDate"].replace("Z", "+00:00"))
        start_hour_utc = dt.hour
        season_year = datetime.strptime(game_date, "%Y-%m-%d").year

        existing = session.query(Game).filter_by(id=game_id).first()
        if not existing:
            game = Game(
                id=game_id,
                date=game_date,
                home_team=home_team,
                away_team=away_team,
                home_score=home_score,
                away_score=away_score,
                start_hour_utc=start_hour_utc,
                timestamp_utc=dt,
                total_runs=total_runs,
                venue=game_data.get("venue", {}).get("name"),
                season_year=season_year,
            )
            session.add(game)
            return game

        # Refresh mutable fields
        existing.home_score = home_score
        existing.away_score = away_score
        existing.total_runs = total_runs
        existing.start_hour_utc = start_hour_utc
        existing.timestamp_utc = dt
        if existing.season_year is None:
            existing.season_year = season_year
        return existing

    def _store_player_stats(self, session: Session, game_id: int, boxscore_data: dict) -> None:
        for team_side in ("home", "away"):
            team_name = boxscore_data["teams"][team_side]["team"]["name"]
            players = boxscore_data["teams"][team_side]["players"]

            for player_id_str, pdata in players.items():
                player_id = int(player_id_str.replace("ID", ""))
                full_name = pdata["person"]["fullName"]
                position = pdata.get("position", {}).get("abbreviation")

                stats = pdata.get("stats", {})
                if not stats:
                    continue
                batting = stats.get("batting", {})
                pitching = stats.get("pitching", {})
                if not batting and not pitching:
                    continue

                # Upsert Player
                player = session.query(Player).filter_by(id=player_id).first()
                if not player:
                    throwing_hand = self._mlb_client.get_player_arm(player_id)
                    player = Player(
                        id=player_id,
                        name=full_name,
                        team=team_name,
                        position=position,
                    )
                    if throwing_hand:
                        player.throwing_hand = throwing_hand
                    session.add(player)
                else:
                    player.name = full_name
                    player.team = team_name
                    player.position = position

                # Upsert PlayerGameStats
                stat_fields = dict(
                    team=team_name,
                    at_bats=batting.get("atBats"),
                    hits=batting.get("hits"),
                    runs=batting.get("runs"),
                    home_runs=batting.get("homeRuns"),
                    rbis=batting.get("rbi"),
                    walks_batting=batting.get("baseOnBalls"),
                    doubles=batting.get("doubles"),
                    triples=batting.get("triples"),
                    innings_pitched=pitching.get("inningsPitched"),
                    outs_pitched=_ip_to_outs(pitching.get("inningsPitched", 0)),
                    earned_runs=pitching.get("earnedRuns"),
                    strikeouts=pitching.get("strikeOuts"),
                    walks=pitching.get("baseOnBalls"),
                    hits_allowed=pitching.get("hits"),
                )
                existing_stat = session.query(PlayerGameStats).filter_by(
                    game_id=game_id, player_id=player_id
                ).first()

                if existing_stat:
                    for k, v in stat_fields.items():
                        setattr(existing_stat, k, v)
                else:
                    session.add(PlayerGameStats(
                        game_id=game_id, player_id=player_id, **stat_fields
                    ))

    def _ingest_sps_for_games(self, session: Session, games: list[Game]) -> None:
        """Fill home/away probable pitcher IDs for any game still missing them."""
        for game in games:
            if game.home_probable_pitcher_id is not None:
                continue
            home_id, away_id = self._mlb_client.get_probable_pitchers(game.id)
            if not home_id or not away_id:
                continue
            pitcher_ids = {home_id, away_id}
            existing_ids = {
                p.id
                for p in session.query(Player.id).filter(Player.id.in_(pitcher_ids)).all()
            }
            if not pitcher_ids.issubset(existing_ids):
                logger.warning(
                    "Game %s: pitchers not in players table yet: %s",
                    game.id, pitcher_ids - existing_ids,
                )
                continue
            game.home_probable_pitcher_id = home_id
            game.away_probable_pitcher_id = away_id
        session.commit()

    # ------------------------------------------------------------------
    # Private – Feature computation
    # ------------------------------------------------------------------

    def _update_features_for_games(
        self, session: Session, games: list[Game], target_date: date
    ) -> None:
        if not games:
            logger.info("No games for %s – skipping feature computation", target_date)
            return

        logger.info("Computing features for %d games on %s", len(games), target_date)

        for game in games:
            home_tf = self._compute_team_features(session, game, game.home_team)
            away_tf = self._compute_team_features(session, game, game.away_team)
            session.merge(home_tf)
            session.merge(away_tf)

            home_pid = getattr(game, "home_probable_pitcher_id", None)
            away_pid = getattr(game, "away_probable_pitcher_id", None)

            if home_pid:
                session.merge(self._compute_pitcher_features(session, game, int(home_pid)))
            else:
                logger.debug("Game %s: no home starter id – skipping home pitcher features", game.id)

            if away_pid:
                session.merge(self._compute_pitcher_features(session, game, int(away_pid)))
            else:
                logger.debug("Game %s: no away starter id – skipping away pitcher features", game.id)

            session.merge(self._compute_bullpen_features(session, game, game.home_team))
            session.merge(self._compute_bullpen_features(session, game, game.away_team))

        session.commit()
        logger.info("Features committed for %s", target_date)

    def _compute_team_features(self, session: Session, game: Game, team: str) -> TeamFeatures:
        last10 = self._get_last_n_games(session, team, game.date, game.season_year, n=10)
        wins_last10 = sum(r.win_flag for r in last10) if last10 else 0
        avg_runs_last10 = (
            float(sum(r.team_runs for r in last10) / max(len(last10), 1))
            if last10 else None
        )
        run_diff = self._get_run_diff_season(session, team, game.date, game.season_year)
        wins, losses = self._get_season_record(session, team, game.date, game.season_year)

        return TeamFeatures(
            team=team,
            game_id=game.id,
            season_year=game.season_year,
            wins_last10=wins_last10,
            avg_runs_last10=avg_runs_last10,
            run_diff_season=run_diff,
            wins_season=wins,
            losses_season=losses,
        )

    def _compute_pitcher_features(
        self, session: Session, game: Game, player_id: int
    ) -> PitcherFeatures:
        totals_sql = text("""
            SELECT
                COALESCE(SUM(pgs.outs_pitched) / 3.0, 0) AS total_ip,
                COALESCE(SUM(pgs.earned_runs), 0)         AS total_er,
                COALESCE(SUM(pgs.walks), 0)               AS total_walks,
                COALESCE(SUM(pgs.hits_allowed), 0)        AS total_hits
            FROM player_game_stats pgs
            JOIN games g ON g.id = pgs.game_id
            WHERE pgs.player_id = :player_id
              AND g.date < :game_date
              AND g.season_year = :season_year
              AND pgs.outs_pitched IS NOT NULL
        """)
        totals = session.execute(totals_sql, {
            "player_id": player_id,
            "game_date": game.date,
            "season_year": game.season_year,
        }).mappings().one()

        total_ip = float(totals["total_ip"] or 0)
        total_er = float(totals["total_er"] or 0)
        total_walks = float(totals["total_walks"] or 0)
        total_hits = float(totals["total_hits"] or 0)

        sp_era = (total_er * 9.0 / total_ip) if total_ip > 0 else None
        sp_whip = ((total_walks + total_hits) / total_ip) if total_ip > 0 else None

        last3_sql = text("""
            SELECT
                COALESCE(SUM(t.earned_runs), 0)         AS er_sum,
                COALESCE(SUM(t.outs_pitched) / 3.0, 0) AS ip_sum
            FROM (
                SELECT pgs.earned_runs, pgs.outs_pitched
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE pgs.player_id = :player_id
                  AND g.date < :game_date
                  AND g.season_year = :season_year
                  AND pgs.outs_pitched IS NOT NULL
                ORDER BY g.date DESC
                LIMIT 3
            ) t
        """)
        last3 = session.execute(last3_sql, {
            "player_id": player_id,
            "game_date": game.date,
            "season_year": game.season_year,
        }).mappings().one()

        last3_ip = float(last3["ip_sum"] or 0)
        last3_er = float(last3["er_sum"] or 0)
        last3_era = (last3_er * 9.0 / last3_ip) if last3_ip > 0 else None

        return PitcherFeatures(
            player_id=player_id,
            game_id=game.id,
            season_year=game.season_year,
            sp_era=sp_era,
            sp_whip=sp_whip,
            last3_era=last3_era,
        )

    def _compute_bullpen_features(
        self, session: Session, game: Game, team: str
    ) -> BullpenFeatures:
        sql = text("""
            WITH starters AS (
                SELECT DISTINCT ON (pgs.game_id, pgs.team)
                    pgs.game_id, pgs.team, pgs.player_id
                FROM player_game_stats pgs
                JOIN games g ON g.id = pgs.game_id
                WHERE g.date < :game_date AND g.season_year = :season_year
                ORDER BY pgs.game_id, pgs.team, pgs.outs_pitched DESC
            )
            SELECT
                COALESCE(SUM(pgs.earned_runs), 0)         AS er_sum,
                COALESCE(SUM(pgs.outs_pitched) / 3.0, 0) AS ip_sum
            FROM player_game_stats pgs
            JOIN games g ON g.id = pgs.game_id
            LEFT JOIN starters s
                ON s.game_id = pgs.game_id
               AND s.team = pgs.team
               AND s.player_id = pgs.player_id
            WHERE pgs.team = :team
              AND g.date < :game_date
              AND g.season_year = :season_year
              AND s.player_id IS NULL
              AND pgs.outs_pitched IS NOT NULL
        """)
        row = session.execute(sql, {
            "team": team,
            "game_date": game.date,
            "season_year": game.season_year,
        }).mappings().one()

        ip_sum = float(row["ip_sum"] or 0)
        er_sum = float(row["er_sum"] or 0)
        bullpen_era = (er_sum * 9.0 / ip_sum) if ip_sum > 0 else None

        return BullpenFeatures(
            team=team,
            game_id=game.id,
            season_year=game.season_year,
            bullpen_era=bullpen_era,
        )

    # ------------------------------------------------------------------
    # Private – SQL aggregate helpers
    # ------------------------------------------------------------------

    def _get_last_n_games(
        self, session: Session, team: str, before_date: date, season_year: int, n: int = 10
    ):
        sql = text("""
            SELECT
                CASE WHEN home_team = :team THEN home_score ELSE away_score END AS team_runs,
                CASE
                    WHEN home_team = :team AND home_score > away_score THEN 1
                    WHEN away_team = :team AND away_score > home_score THEN 1
                    ELSE 0
                END AS win_flag
            FROM games
            WHERE (home_team = :team OR away_team = :team)
              AND date < :before_date
              AND season_year = :season_year
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY date DESC
            LIMIT :n
        """)
        return session.execute(sql, {
            "team": team,
            "before_date": before_date,
            "season_year": season_year,
            "n": n,
        }).fetchall()

    def _get_season_record(
        self, session: Session, team: str, before_date: date, season_year: int
    ) -> tuple[int, int]:
        sql = text("""
            SELECT
                SUM(CASE
                    WHEN home_team = :team AND home_score > away_score THEN 1
                    WHEN away_team = :team AND away_score > home_score THEN 1
                    ELSE 0
                END) AS wins,
                SUM(CASE
                    WHEN home_team = :team AND home_score < away_score THEN 1
                    WHEN away_team = :team AND away_score < home_score THEN 1
                    ELSE 0
                END) AS losses
            FROM games
            WHERE (home_team = :team OR away_team = :team)
              AND date < :before_date
              AND season_year = :season_year
              AND home_score IS NOT NULL AND away_score IS NOT NULL
        """)
        row = session.execute(sql, {
            "team": team,
            "before_date": before_date,
            "season_year": season_year,
        }).fetchone()
        return (int(row.wins or 0), int(row.losses or 0))

    def _get_run_diff_season(
        self, session: Session, team: str, before_date: date, season_year: int
    ) -> int:
        sql = text("""
            SELECT COALESCE(SUM(
                CASE
                    WHEN home_team = :team THEN home_score - away_score
                    WHEN away_team = :team THEN away_score - home_score
                    ELSE 0
                END
            ), 0) AS run_diff
            FROM games
            WHERE (home_team = :team OR away_team = :team)
              AND date < :before_date
              AND season_year = :season_year
              AND home_score IS NOT NULL AND away_score IS NOT NULL
        """)
        row = session.execute(sql, {
            "team": team,
            "before_date": before_date,
            "season_year": season_year,
        }).scalar_one_or_none()
        return int(row or 0)

    # ------------------------------------------------------------------
    # Public – Weather backfill
    # ------------------------------------------------------------------

    def backfill_missing_weather(self, batch_size: int = 10) -> dict:
        """Fetch weather for every historical game that is missing a record.

        Queries all games without a Weather row, ordered oldest-first, and
        fills them in until either every game is covered or the weather API
        credit limit is exhausted.  Progress is committed every `batch_size`
        games so a mid-run stop doesn't lose work.

        Parameters
        ----------
        batch_size : int
            Number of games to commit in one transaction (default 10).

        Returns
        -------
        dict with keys:
            filled        – games that received a new weather record
            skipped       – games skipped (no venue mapping or missing time)
            stopped_early – True if the API credit limit was hit
        """
        summary = {"filled": 0, "skipped": 0, "stopped_early": False}

        with SessionLocal() as session:
            games_missing_weather = (
                session.query(Game)
                .filter(~exists().where(Weather.game_id == Game.id))
                .order_by(Game.date.asc())
                .all()
            )

            logger.info(
                "backfill_missing_weather: %d games without weather data",
                len(games_missing_weather),
            )

            for game in games_missing_weather:
                try:
                    reading = self._resolve_weather(game)
                except WeatherApiLimitError:
                    logger.warning(
                        "Weather API limit reached – stopping after %d filled", summary["filled"]
                    )
                    session.commit()
                    summary["stopped_early"] = True
                    return summary

                if reading is None:
                    summary["skipped"] += 1
                    continue

                session.add(Weather(
                    game_id=game.id,
                    temperature=reading["temperature"],
                    wind_speed=reading["wind_speed"],
                    wind_direction=str(reading["wind_direction"]),
                    precipitation=reading["precipitation"] > 0,
                ))
                summary["filled"] += 1
                logger.debug("Weather filled for game %s at %s", game.id, game.venue)

                if summary["filled"] % batch_size == 0:
                    session.commit()
                    logger.info("Weather backfill progress: %d filled so far", summary["filled"])

            session.commit()

        logger.info(
            "backfill_missing_weather complete: %d filled, %d skipped",
            summary["filled"], summary["skipped"],
        )
        return summary

    # ------------------------------------------------------------------
    # Private – Weather
    # ------------------------------------------------------------------

    def _resolve_weather(self, game: Game):
        """Resolve city/hour for a game and call the weather client.

        Returns a WeatherReading, or None if the game's venue or start time
        can't be mapped.  Propagates WeatherApiLimitError to the caller.
        """
        city = _VENUE_TO_CITY.get(game.venue)
        if not city or game.start_hour_utc is None or game.venue not in venue_utc_offsets:
            return None
        local_hour = (game.start_hour_utc + venue_utc_offsets[game.venue]) % 24
        return self._weather_client.get_hourly_weather(city, game.date, local_hour)

    def _ingest_weather_for_games(self, session: Session, games: list[Game]) -> None:
        """Add weather records for games on a single ingested day."""
        for game in games:
            existing = session.query(Weather).filter_by(game_id=game.id).first()
            if existing:
                continue  # historical weather doesn't change

            try:
                reading = self._resolve_weather(game)
            except WeatherApiLimitError:
                logger.warning("Weather API limit reached – stopping early for this day")
                session.commit()
                break

            if not reading:
                continue

            session.add(Weather(
                game_id=game.id,
                temperature=reading["temperature"],
                wind_speed=reading["wind_speed"],
                wind_direction=str(reading["wind_direction"]),
                precipitation=reading["precipitation"] > 0,
            ))
            logger.debug("Weather added for game %s at %s", game.id, game.venue)

        session.commit()
