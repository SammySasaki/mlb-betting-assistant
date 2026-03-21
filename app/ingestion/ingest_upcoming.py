"""
app/ingestion/ingest_upcoming.py

CLI entry-point for pre-game data ingestion (today's games, lineups, pitchers).
Run this before games start; run backfill_mlb_games.py afterwards to fill scores.

Usage:
    python ingest_upcoming.py
"""
from datetime import date
from app.services.ingestion_service import MLBIngestionService

if __name__ == "__main__":
    service = MLBIngestionService()
    summary = service.ingest_upcoming_day(date.today())
    print(f"Done: {summary['games_stored']} game(s) stored for {summary['date']}.")
    if summary["errors"]:
        for err in summary["errors"]:
            print(f"  ERROR: {err}")
