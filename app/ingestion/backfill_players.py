import requests
import sys
from datetime import date, timedelta, datetime
from sqlalchemy.orm import Session
from infra.db.init_db import SessionLocal
from app.db_models import Game, Player, PlayerGameStats



session: Session = SessionLocal()

def ingest_players():
    players = session.query(Player).all()
    for player in players:
        throwing_hand = get_player_arm(player.id)
        if throwing_hand:
            player.throwing_hand = throwing_hand
    session.commit()


player_arm_cache = {}

def get_player_arm(player_id):
    if player_id in player_arm_cache:
        return player_arm_cache[player_id]
    
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/"
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch player {player_id}: {response.status_code}")
        return None

    try:
        data = response.json()
        code = data["people"][0]["pitchHand"]["code"]
        player_arm_cache[player_id] = code
        return code
    except (KeyError, IndexError, ValueError) as e:
        print(f"Error parsing response for player {player_id}: {e}")
        return None


if __name__ == "__main__":
    ingest_players()
    session.close()