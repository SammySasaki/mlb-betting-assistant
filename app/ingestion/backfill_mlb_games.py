import requests
import sys
from datetime import date, timedelta, datetime
from sqlalchemy.orm import Session
from sqlalchemy import text
from infra.db.init_db import SessionLocal
from app.db_models import Game, Player, PlayerGameStats
from backfill_players import get_player_arm
from backfill_weather import backfill_weather
from backfill_sps import ingest_sps
from app.db_models import TeamFeatures, PitcherFeatures, BullpenFeatures
from app.services.lineup_service import LineupService
from app.implementations.sqlalchemy_lineup_repository import SQLAlchemyLineupRepository
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.mlb_api_client import StatsApiClient



session: Session = SessionLocal()

session = SessionLocal()
player_stats_repo = SqlAlchemyPlayerGameStatsRepository(session)
lineup_repo = SQLAlchemyLineupRepository(session)
game_repo = GameRepository(session)
mlb_client = StatsApiClient()

lineup_service = LineupService(
    player_stats_repo=player_stats_repo,
    lineup_repo=lineup_repo,
    mlb_client=mlb_client,
    game_repo=game_repo,
)

def get_mlb_games_for_date(target_date):
    url = f"https://statsapi.mlb.com/api/v1/schedule"
    params = {
        "sportId": 1,
        "date": target_date.strftime("%Y-%m-%d"),
        "hydrate": "team,linescore"
    }

    response = requests.get(url, params=params)
    data = response.json()
    return data.get("dates", [])[0].get("games", []) if data.get("dates") else []

def get_boxscore(game_id):
    url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
    response = requests.get(url)
    return response.json()

player_cache = {}

def get_player(player_id):
    if player_id in player_cache:
        return player_cache[player_id]
    
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/"
    response = requests.get(url)
    data = response.json()
    
    player_cache[player_id] = data
    return data


def store_game(game_data):
    game_id = game_data["gamePk"]
    game_date = game_data["officialDate"]
    home_team = game_data["teams"]["home"]["team"]["name"]
    away_team = game_data["teams"]["away"]["team"]["name"]
    try:
        home_score = game_data["teams"]["home"]["score"]
        away_score = game_data["teams"]["away"]["score"]
    except KeyError:
        # Skip game if scores are not available (e.g., game hasn't been played yet)
        return None
    total_runs = home_score + away_score
    dt = datetime.fromisoformat(game_data["gameDate"].replace("Z", "+00:00"))
    start_hour_utc = dt.hour
    game = Game(
        id=game_id,
        date=game_date,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        start_hour_utc=start_hour_utc,
        total_runs=total_runs,
        venue=game_data.get("venue", {}).get("name", None),
        season_year=datetime.strptime(game_date, "%Y-%m-%d").year
    )

    # Insert if not already there
    existing = session.query(Game).filter_by(id=game_id).first()
    if not existing:
        session.add(game)
    else:
        # Update only if values have changed or are newly available
        if existing.home_score != home_score:
            existing.home_score = home_score
        if existing.away_score != away_score:
            existing.away_score = away_score
        if existing.total_runs != total_runs:
            existing.total_runs = total_runs
        if existing.start_hour_utc is None and start_hour_utc is not None:
            existing.start_hour_utc = start_hour_utc
        if existing.season_year is None:
            existing.season_year = datetime.strptime(game_date, "%Y-%m-%d").year
    return game

def store_player_stats(game_id, boxscore_data):
    teams = ["home", "away"]
    for team_side in teams:
        team_name = boxscore_data["teams"][team_side]["team"]["name"]
        players = boxscore_data["teams"][team_side]["players"]

        for player_id_str, pdata in players.items():
            player_id = int(player_id_str.replace("ID", ""))
            full_name = pdata["person"]["fullName"]
            position = pdata.get("position", {}).get("abbreviation", None)

        
            # Stats
            stats = pdata.get("stats", {})
            if not stats:
                continue

            batting = stats.get("batting", {})
            pitching = stats.get("pitching", {})

            if pitching == {} and batting == {}:
                continue
            

            # Create or get Player
            player = session.query(Player).filter_by(id=player_id).first()
            if not player:
                throwing_hand = get_player_arm(player_id)
                player = Player(id=player_id, name=full_name, team=team_name, position=position)
                if throwing_hand:
                    player.throwing_hand = throwing_hand
                session.add(player)
            else:
                # Update fields in case they changed
                player.name = full_name  # optional if you want to refresh name
                player.team = team_name
                player.position = position

            def ip_to_outs(ip: float) -> int:
                """
                Convert MLB innings pitched (X, X.1, X.2) into total outs.
                Example: 6.1 -> 19 outs, 6.2 -> 20 outs.
                """
                if ip is None:
                    return 0
                try:
                    ip_float = float(ip)
                except (ValueError, TypeError):
                    return 0
                innings = int(ip_float)
                fraction = round((ip_float - innings) * 10)  # .1 -> 1, .2 -> 2
                if fraction not in (0, 1, 2):
                    raise ValueError(f"Invalid innings format: {ip}")
                return innings * 3 + fraction
            player_stat = PlayerGameStats(
                game_id=game_id,
                player_id=player_id,
                team=team_name,
                at_bats=batting.get("atBats"),
                hits=batting.get("hits"),
                runs=batting.get("runs"),
                home_runs=batting.get("homeRuns"),
                rbis=batting.get("rbi"),
                walks_batting=batting.get("baseOnBalls"),
                doubles=batting.get("doubles"),
                triples=batting.get("triples"),
                innings_pitched=pitching.get("inningsPitched", 0),
                outs_pitched=ip_to_outs(pitching.get("inningsPitched", 0)),
                earned_runs=pitching.get("earnedRuns"),
                strikeouts=pitching.get("strikeOuts"),
                walks=pitching.get("baseOnBalls"),
                hits_allowed=pitching.get("hits")
            )

            # Avoid duplicates
            existing = session.query(PlayerGameStats).filter_by(
                game_id=game_id, player_id=player_id
            ).first()


            if existing:
                existing.team = team_name
                existing.at_bats = batting.get("atBats")
                existing.hits = batting.get("hits")
                existing.runs = batting.get("runs")
                existing.home_runs = batting.get("homeRuns")
                existing.rbis = batting.get("rbi")
                existing.walks_batting = batting.get("baseOnBalls")
                existing.doubles = batting.get("doubles")
                existing.triples = batting.get("triples")
                existing.innings_pitched = pitching.get("inningsPitched")
                existing.earned_runs = pitching.get("earnedRuns")
                existing.strikeouts = pitching.get("strikeOuts")
                existing.walks = pitching.get("baseOnBalls")
                existing.hits_allowed = pitching.get("hits")
            else:
                session.add(player_stat)

def backfill_games_with_stats(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        print(f"\nFetching games for {current}")
        try:
            games = get_mlb_games_for_date(current)
            for g in games:
                # Regular season
                if g["gameType"] == "R":
                    game = store_game(g)
                    if game:
                        boxscore = get_boxscore(g["gamePk"])
                        store_player_stats(game.id, boxscore)
            lineup_service.save_lineups_by_date(current)
            session.commit()
        except Exception as e:
            session.rollback()
            print(f"Error on {current}: {e}")
        current += timedelta(days=1)

### --- low-level SQL helpers (use text() for heavier aggregates) --- ###
def _get_last_n_games(session: Session, team: str, before_date: date, season_year: int, n: int = 10):
    """
    Return rows (team_runs INT, win_flag INT) for the last n games for `team` BEFORE before_date.
    Uses SQL so the DB does the filtering and limiting.
    """
    sql = text(
        """
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
        """
    )
    rows = session.execute(sql, {"team": team, "before_date": before_date, "season_year": season_year, "n": n}).fetchall()
    return rows

def _get_season_record(session: Session, team: str, before_date: date, season_year: int):
    """
    Return (wins, losses) for the team up to `before_date` in the given season.
    """
    sql = text(
        """
        SELECT
            SUM(
                CASE
                    WHEN home_team = :team AND home_score > away_score THEN 1
                    WHEN away_team = :team AND away_score > home_score THEN 1
                    ELSE 0
                END
            ) AS wins,
            SUM(
                CASE
                    WHEN home_team = :team AND home_score < away_score THEN 1
                    WHEN away_team = :team AND away_score < home_score THEN 1
                    ELSE 0
                END
            ) AS losses
        FROM games
        WHERE (home_team = :team OR away_team = :team)
          AND date < :before_date
          AND season_year = :season_year
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        """
    )
    row = session.execute(sql, {"team": team, "before_date": before_date, "season_year": season_year}).fetchone()
    return (row.wins or 0, row.losses or 0)


def _get_run_diff_season(session: Session, team: str, before_date: date, season_year: int) -> int:
    """Season-to-date run differential (up to but excluding before_date)."""
    sql = text(
        """
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
        """
    )
    row = session.execute(sql, {"team": team, "before_date": before_date, "season_year": season_year}).scalar_one_or_none()
    return int(row or 0)


def compute_team_features(session: Session, game: Game, team: str) -> TeamFeatures:
    """
    Compute TeamFeatures for `team` (either home or away) for the supplied game.
    Returns a TeamFeatures ORM instance (not yet committed).
    """
    last10 = _get_last_n_games(session, team, game.date, game.season_year, n=10)
    wins_last10 = sum(r.win_flag for r in last10) if last10 else 0
    avg_runs_last10 = float(sum(r.team_runs for r in last10) / max(len(last10), 1)) if last10 else None

    run_diff = _get_run_diff_season(session, team, game.date, game.season_year)

    season_wins, season_losses = _get_season_record(session, team, game.date, game.season_year)


    return TeamFeatures(
        team=team,
        game_id=game.id,
        season_year=game.season_year,
        wins_last10=wins_last10,
        avg_runs_last10=avg_runs_last10,
        run_diff_season=run_diff,
        wins_season=season_wins,
        losses_season=season_losses
    )


def compute_pitcher_features(session: Session, game: Game, player_id: int) -> PitcherFeatures:
    """
    Compute pitcher-level aggregates before this game:
      - season totals (IP, ER, K, BB, hits)
      - sp_era, sp_whip
      - last3_era (ERA over last 3 starts)
    Requires `player_game_stats` table to exist with relevant columns.
    """
    # aggregate totals prior to game date in same season
    totals_sql = text(
        """
        SELECT
            COALESCE(SUM(pgs.outs_pitched) / 3.0, 0) AS total_ip,
            COALESCE(SUM(pgs.earned_runs), 0) AS total_er,
            COALESCE(SUM(pgs.strikeouts), 0) AS total_k,
            COALESCE(SUM(pgs.walks), 0) AS total_walks,
            COALESCE(SUM(pgs.hits_allowed), 0) AS total_hits
        FROM player_game_stats pgs
        JOIN games g ON g.id = pgs.game_id
        WHERE pgs.player_id = :player_id
          AND g.date < :game_date
          AND g.season_year = :season_year
          AND pgs.outs_pitched IS NOT NULL
        """
    )
    totals = session.execute(totals_sql, {"player_id": player_id, "game_date": game.date, "season_year": game.season_year}).mappings().one()

    total_ip = float(totals["total_ip"]) if totals["total_ip"] is not None else 0.0
    total_er = float(totals["total_er"]) if totals["total_er"] is not None else 0.0
    total_walks = float(totals["total_walks"]) if totals["total_walks"] is not None else 0.0
    total_hits = float(totals["total_hits"]) if totals["total_hits"] is not None else 0.0

    sp_era = (total_er * 9.0 / total_ip) if total_ip > 0 else None
    sp_whip = ((total_walks + total_hits) / total_ip) if total_ip > 0 else None

    # last 3 starts ERA
    last3_sql = text(
        """
        SELECT COALESCE(SUM(t.earned_runs),0) AS er_sum, COALESCE(SUM(t.outs_pitched) / 3.0, 0) AS ip_sum FROM (
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
        """
    )
    last3 = session.execute(last3_sql, {"player_id": player_id, "game_date": game.date, "season_year": game.season_year}).mappings().one()
    last3_ip = float(last3["ip_sum"]) if last3["ip_sum"] is not None else 0.0
    last3_er = float(last3["er_sum"]) if last3["er_sum"] is not None else 0.0
    last3_era = (last3_er * 9.0 / last3_ip) if last3_ip > 0 else None

    return PitcherFeatures(
        player_id=player_id,
        game_id=game.id,
        season_year=game.season_year,
        sp_era=sp_era,
        sp_whip=sp_whip,
        last3_era=last3_era,
    )


def compute_bullpen_features(session: Session, game: Game, team: str) -> BullpenFeatures:
    """
    Compute bullpen ERA for team up to but excluding the current game date.
    We treat 'starter' per historical game as the pitcher with max outs_pitched (DISTINCT ON),
    and then sum earned runs / innings for non-starters.
    """
    sql = text(
        """
        WITH starters AS (
          SELECT DISTINCT ON (pgs.game_id, pgs.team)
            pgs.game_id, pgs.team, pgs.player_id
          FROM player_game_stats pgs
          JOIN games g ON g.id = pgs.game_id
          WHERE g.date < :game_date AND g.season_year = :season_year
          ORDER BY pgs.game_id, pgs.team, pgs.outs_pitched DESC
        )
        SELECT
          COALESCE(SUM(pgs.earned_runs), 0) AS er_sum,
          COALESCE(SUM(pgs.outs_pitched) / 3.0, 0) AS ip_sum
        FROM player_game_stats pgs
        JOIN games g ON g.id = pgs.game_id
        LEFT JOIN starters s ON s.game_id = pgs.game_id AND s.team = pgs.team AND s.player_id = pgs.player_id
        WHERE pgs.team = :team
          AND g.date < :game_date
          AND g.season_year = :season_year
          AND s.player_id IS NULL
          AND pgs.outs_pitched IS NOT NULL
        """
    )
    row = session.execute(sql, {"team": team, "game_date": game.date, "season_year": game.season_year}).mappings().one()
    ip_sum = float(row["ip_sum"]) if row["ip_sum"] is not None else 0.0
    er_sum = float(row["er_sum"]) if row["er_sum"] is not None else 0.0
    bullpen_era = (er_sum * 9.0 / ip_sum) if ip_sum > 0 else None

    return BullpenFeatures(
        team=team,
        game_id=game.id,
        season_year=game.season_year,
        bullpen_era=bullpen_era,
    )


def update_features_for_day(target_date: date):
    """
    Compute and upsert features for all games on target_date.
    Uses a fresh session and commits per day (idempotent via session.merge()).
    """
    with SessionLocal() as session:
        games = session.query(Game).filter(Game.date == target_date).all()
        if not games:
            print(f"No games found for {target_date}")
            return

        print(f"Updating features for {len(games)} games on {target_date}")

        for game in games:
            # Team features (home + away)
            try:
                home_tf = compute_team_features(session, game, team=game.home_team)
                away_tf = compute_team_features(session, game, team=game.away_team)
                session.merge(home_tf)
                session.merge(away_tf)
            except Exception as e:
                session.rollback()
                print(f"ERROR computing team features for game {game.id}: {e}")
                raise

            # Pitcher features (if probable starters are present)
            try:
                # you'll likely have columns home_probable_pitcher_id / away_probable_pitcher_id
                home_pid = getattr(game, "home_probable_pitcher_id", None) or getattr(game, "home_starter_id", None)
                away_pid = getattr(game, "away_probable_pitcher_id", None) or getattr(game, "away_starter_id", None)

                if home_pid:
                    pf_home = compute_pitcher_features(session, game, int(home_pid))
                    session.merge(pf_home)
                else:
                    print(f"  no home starter id for game {game.id}; skipping home pitcher features")

                if away_pid:
                    pf_away = compute_pitcher_features(session, game, int(away_pid))
                    session.merge(pf_away)
                else:
                    print(f"  no away starter id for game {game.id}; skipping away pitcher features")
            except Exception as e:
                session.rollback()
                print(f"ERROR computing pitcher features for game {game.id}: {e}")
                raise

            # Bullpen features
            try:
                bf_home = compute_bullpen_features(session, game, game.home_team)
                bf_away = compute_bullpen_features(session, game, game.away_team)
                session.merge(bf_home)
                session.merge(bf_away)
            except Exception as e:
                session.rollback()
                print(f"ERROR computing bullpen features for game {game.id}: {e}")
                raise

        # commit all rows for the day
        session.commit()
        print(f"Committed features for games on {target_date}")


def backfill_features(start_date: date, end_date: date, batch_days: int = 7):
    """
    Backfill a date range. Uses chunking to avoid runaway transactions.
    Default: process 7 days at a time (commits per day inside update_features_for_day).
    """
    cur = start_date
    while cur <= end_date:
        print(f"Backfilling {cur}")
        update_features_for_day(cur)
        cur += timedelta(days=1)


if __name__ == "__main__":
    
    # Default date range
    START = date(2025, 9, 17)
    # START = date(2021, 3, 7)
    END = date(2025, 9, 20)
    # END = date(2022, 4, 7)

    # Check for CLI arg
    if len(sys.argv) > 1 and sys.argv[1].lower() == "yesterday":
        yesterday = date.today() - timedelta(days=1)
        START = END = yesterday

    print(f"Running stats backfill from {START} to {END}")
    backfill_games_with_stats(START, END)
    update_features_for_day(date.today())
    backfill_weather()
    ingest_sps()



    # start = date(2025, 9, 20)
    # # start = date(2025, 9, 16)
    # end = date.today()
    # backfill_features(start, end)

    session.close()




