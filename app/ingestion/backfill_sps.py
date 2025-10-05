import requests
import sys
from datetime import date, timedelta, datetime
from sqlalchemy.orm import Session
from sqlalchemy import null
from infra.db.init_db import SessionLocal
from app.db_models import Game, Player, PlayerGameStats



session: Session = SessionLocal()

def ingest_sps():
    games = session.query(Game).filter(Game.home_probable_pitcher_id == None).all()

    for g in games:
        try:
            home_id, away_id = get_probable_pitchers(g.id)

            # Check if both pitcher IDs exist
            pitcher_ids = {home_id, away_id}
            existing_ids = {
                p.id for p in session.query(Player.id).filter(Player.id.in_(pitcher_ids)).all()
            }

            if not pitcher_ids.issubset(existing_ids):
                print(f"[SKIP] Game {g.id}: Missing pitchers in players table: {pitcher_ids - existing_ids}")
                continue

            g.home_probable_pitcher_id = home_id
            g.away_probable_pitcher_id = away_id
            session.commit()

        except Exception as e:
            session.rollback()

    session.close()

def get_probable_pitchers(game_id: str):
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePk={game_id}&hydrate=probablePitcher"
    response = requests.get(url).json()
    return (response["dates"][0]["games"][0]["teams"]["home"]["probablePitcher"]["id"], response["dates"][0]["games"][0]["teams"]["away"]["probablePitcher"]["id"])


if __name__ == "__main__":
    ingest_sps()
    session.close()