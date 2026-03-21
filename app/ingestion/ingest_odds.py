"""
app/ingestion/ingest_odds.py

CLI entry-point for odds ingestion.

Usage:
    # Ingest current lines for today's games:
    python ingest_odds.py

    # Backfill historical totals for a past date:
    python ingest_odds.py 2025-09-20
"""
import sys
from datetime import date

from app.services.odds_service import OddsService

if __name__ == "__main__":
    service = OddsService()

    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
        result = service.ingest_historical_lines_for_date(target)
        print(f"Historical lines for {result['date']}: {result['updated']} updated.")
    else:
        today = date.today()
        result = service.ingest_lines_for_date(today)
        print(
            f"Lines for {result['date']}: "
            f"{result['totals_updated']} totals, {result['ml_updated']} ML updated."
        )

    if result.get("errors"):
        for err in result["errors"]:
            print(f"  ERROR: {err}")
