"""
app/ingestion/backfill_mlb_games.py

CLI entry-point for MLB data ingestion.

Usage
-----
    # Backfill a fixed range (edit START / END below):
    python backfill_mlb_games.py

    # Re-ingest only yesterday:
    python backfill_mlb_games.py yesterday
"""

import sys
from datetime import date, timedelta

from app.services.ingestion_service import MLBIngestionService

if __name__ == "__main__":
    START = date(2025, 9, 17)
    END = date(2025, 9, 20)

    if len(sys.argv) > 1 and sys.argv[1].lower() == "yesterday":
        yesterday = date.today() - timedelta(days=1)
        START = END = yesterday

    service = MLBIngestionService()
    summary = service.ingest_dates(START, END)

    print(f"\nDone: {summary['total_games']} games across {len(summary['days'])} day(s).")
    if summary["total_errors"]:
        print(f"  {summary['total_errors']} error(s) occurred – check logs for details.")
        for day in summary["days"]:
            for err in day["errors"]:
                print(f"  [{day['date']}] {err}")
