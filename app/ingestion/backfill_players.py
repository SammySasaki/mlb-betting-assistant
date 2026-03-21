"""
app/ingestion/backfill_players.py

One-time admin script: backfills throwing_hand for every Player row that
doesn't have one yet.  Run manually when new players are missing arm data.

Usage:
    python backfill_players.py
"""
from infra.db.init_db import SessionLocal
from app.db.models import Player
from app.implementations.mlb_api_client import StatsApiClient


def ingest_players() -> None:
    client = StatsApiClient()
    with SessionLocal() as session:
        players = session.query(Player).filter(Player.throwing_hand == None).all()
        print(f"Backfilling throwing_hand for {len(players)} players...")
        for player in players:
            throwing_hand = client.get_player_arm(player.id)
            if throwing_hand:
                player.throwing_hand = throwing_hand
        session.commit()
        print("Done.")


if __name__ == "__main__":
    ingest_players()
