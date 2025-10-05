import requests
import sys
from sqlalchemy.orm import Session
from app.db_models import Game, Player
from infra.db.init_db import SessionLocal
from datetime import datetime, timedelta
from app.services.lineup_service import LineupService
from app.implementations.sqlalchemy_lineup_repository import SQLAlchemyLineupRepository
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.mlb_api_client import StatsApiClient

def fetch_mlb_results(result_date):
    print("fetching for " + str(result_date))
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={result_date}&expand=schedule.linescore"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()["dates"][0]["games"]

def get_probable_pitchers(game_id: str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePk={game_id}&hydrate=probablePitcher"
    response = requests.get(url).json()
    return (response["dates"][0]["games"][0]["teams"]["home"]["probablePitcher"]["id"], response["dates"][0]["games"][0]["teams"]["away"]["probablePitcher"]["id"])


def get_or_create_player(db: Session, player_id: int, team: str = None):
    player = db.query(Player).filter_by(id=player_id).first()
    if not player:
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        if "people" not in data or not data["people"]:
            return None

        person = data["people"][0]
        code = person["pitchHand"]["code"]
        print(person)
        player = Player(
            id=person["id"],
            name=person.get("fullName"),
            team=team,
            position=person.get("primaryPosition", {}).get("abbreviation"),
            throwing_hand=code
        )
        db.add(player)
        db.flush()  # assign ID before referencing it
    else:
        # Update team, position, and throwing_hand if player exists
        url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        if "people" in data and data["people"]:
            person = data["people"][0]
            player.team = team
            player.position = person.get("primaryPosition", {}).get("abbreviation")
            player.throwing_hand = person["pitchHand"]["code"]
            db.flush()
    return player

def update_results(result_date):
    db: Session = SessionLocal()
    games = fetch_mlb_results(result_date)
    print("ingesting " + str(len(games)) + " games for " + str(result_date))
    for game_data in games:
        if game_data["gameType"] == "R":
            try:
                game_id = game_data["gamePk"]
                game_date = game_data["officialDate"]
                home_team = game_data["teams"]["home"]["team"]["name"]
                away_team = game_data["teams"]["away"]["team"]["name"]
                dt = datetime.fromisoformat(game_data["gameDate"].replace("Z", "+00:00"))
                start_hour_utc = dt.hour
                home_probable_pitcher_id, away_probable_pitcher_id = get_probable_pitchers(game_id)

                if home_probable_pitcher_id:
                    get_or_create_player(db, home_probable_pitcher_id, home_team)
                if away_probable_pitcher_id:
                    get_or_create_player(db, away_probable_pitcher_id, away_team)

                # check existing Game
                game = db.query(Game).filter_by(id=game_id).first()
                if game:
                    print("game exists")
                else:
                    game = Game(
                        id=game_id,
                        date=game_date,
                        home_team=home_team,
                        away_team=away_team,
                        start_hour_utc=start_hour_utc,
                        home_probable_pitcher_id=home_probable_pitcher_id,
                        away_probable_pitcher_id=away_probable_pitcher_id,
                        venue=game_data.get("venue", {}).get("name", None),
                        season_year=datetime.strptime(game_date, "%Y-%m-%d").year
                    )
                    db.add(game)
            except Exception as e:
                print(f"Error updating game result: {e}")
    db.commit()
    db.close()

if __name__ == "__main__":
    session = SessionLocal()
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
    today = datetime.today().date()
    update_results(today)
    service.save_lineups_by_date(today)
    
    # tomorrow = today - timedelta(days=1)
    # update_results(tomorrow)


