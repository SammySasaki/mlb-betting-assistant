from sqlalchemy.orm import Session
from sqlalchemy import delete
from datetime import datetime
from app.db_models import Game
from infra.db.init_db import engine

REGULAR_SEASON_START_DATES = {
    2022: datetime(2022, 4, 7),
    2023: datetime(2023, 3, 30),
    2024: datetime(2024, 3, 28),
    2025: datetime(2025, 3, 27),
}

ALLSTAR_GAME_DATES = {
    2022: datetime(2022, 7, 19),
    2023: datetime(2023, 7, 11),
    2024: datetime(2024, 7, 16),
    2025: datetime(2025, 7, 15),
}


def delete_preseason_games():
    with Session(engine) as session:
        for season, start_date in REGULAR_SEASON_START_DATES.items():
            preseason_games = session.query(Game).filter(
                Game.season_year == season,
                Game.date < start_date
            ).all()
            
            print(f"Deleting {len(preseason_games)} preseason games from {season}...")

            for game in preseason_games:
                session.delete(game)  # cascades if set up properly

        session.commit()
        print("Preseason games removed.")

def delete_all_star_games():
    with Session(engine) as session:
        for season, as_date in ALLSTAR_GAME_DATES.items():
            as_games = session.query(Game).filter(
                Game.season_year == season,
                Game.date == as_date
            ).all()
            
            print(f"Deleting {len(as_games)} all star games from {season}...")

            for game in as_games:
                session.delete(game)  # cascades if set up properly

        session.commit()
        print("Preseason games removed.")

if __name__ == "__main__":
    # delete_preseason_games()
    delete_all_star_games()