import requests
import pandas as pd
import os
from sqlalchemy import create_engine, text
from difflib import SequenceMatcher

API_KEY = os.getenv("THE_ODDS_API_KEY")
SPORT = "baseball_mlb"
REGION = "us" 
MARKET = "totals"
BOOKMAKER_KEY = "hardrockbet"

# --- DB Connection ---
from infra.db.init_db import engine

# --- Step 1: Get upcoming games from DB ---
def fetch_upcoming_games():
    with engine.connect() as conn:
        query = """
        SELECT id, date, home_team, away_team
        FROM games
        WHERE date >= CURRENT_DATE AND total_runs IS NULL
        """
        return pd.read_sql(query, conn).to_dict(orient="records")
    
def fetch_games_by_date(game_date: str):
    """
    Fetch games from DB for a specific date (YYYY-MM-DD).
    """
    with engine.connect() as conn:
        query = text("""
            SELECT id, date, home_team, away_team
            FROM games
            WHERE date::date = :game_date
        """)
        return pd.read_sql(query, conn, params={"game_date": game_date}).to_dict(orient="records")

# --- Step 2: Get odds from The Odds API ---
def fetch_totals_odds():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "regions": REGION,
        "markets": MARKET,
        "bookmakers": BOOKMAKER_KEY,
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_historical_totals_odds(game_date: str):
    """
    Fetch totals odds for a historical date from The Odds API.
    """
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds-history"
    params = {
        "regions": REGION,
        "markets": MARKET,
        "bookmakers": BOOKMAKER_KEY,
        "apiKey": API_KEY,
        "date": f"{game_date}T00:00:00Z",  # ISO8601 format required
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def fetch_ml_odds():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "regions": REGION,
        "markets": "h2h",
        "bookmakers": BOOKMAKER_KEY,
        "apiKey": API_KEY,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def team_names_match(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """
    Returns True if team names are a fuzzy or partial match.
    Handles cases like 'Athletics' vs 'Oakland Athletics'.
    """
    name1 = name1.lower()
    name2 = name2.lower()

    # Exact or fuzzy match
    if name1 == name2 or SequenceMatcher(None, name1, name2).ratio() >= threshold:
        return True

    # Substring match
    if name1 in name2 or name2 in name1:
        return True

    return False

# --- Step 3: Match games + update DB ---
def update_game_moneylines():
    upcoming_games = fetch_upcoming_games()
    odds_data = fetch_ml_odds()

    with engine.begin() as conn:
        for game in upcoming_games:
            home = game["home_team"].lower()
            away = game["away_team"].lower()
            game_id = game["id"]

            match_found = None

            for match in odds_data:
                match_home = match["home_team"]
                match_away = match["away_team"]
                if team_names_match(home, match_home) and team_names_match(away, match_away):
                    bookmaker = next(
                        (b for b in match["bookmakers"] if b["key"] == "hardrockbet"), None
                    )
                    if not bookmaker:
                        continue

                    ml_market = next(
                        (m for m in bookmaker["markets"] if m["key"] == "h2h"), None
                    )
                    if not ml_market:
                        continue

                    home_price = None
                    away_price = None

                    for outcome in ml_market["outcomes"]:
                        if outcome["name"] == match_home:
                            home_price = outcome["price"]
                        elif outcome["name"] == match_away:
                            away_price = outcome["price"]

                    if home_price is not None and away_price is not None:
                        conn.execute(
                            text("""
                                UPDATE games
                                SET 
                                    home_ml_price = :home_price,
                                    away_ml_price = :away_price
                                WHERE id = :game_id
                            """),
                            {
                                "home_price": home_price,
                                "away_price": away_price,
                                "game_id": game_id,
                            }
                        )
                        print(f"✅ Updated game {game_id} with lines home {home_price}, away {away_price}")
                        match_found = True
                        break

            if not match_found:
                print(f"❌ No line found for {away} @ {home}")

def update_games_with_historical_lines(game_date: str):
    games = fetch_games_by_date(game_date)
    odds_data = fetch_historical_totals_odds(game_date)

    with engine.begin() as conn:
        for game in games:
            home = game["home_team"].lower()
            away = game["away_team"].lower()
            game_id = game["id"]

            match_found = None

            for match in odds_data:
                match_home = match["home_team"]
                match_away = match["away_team"]

                if team_names_match(home, match_home) and team_names_match(away, match_away):
                    bookmaker = next(
                        (b for b in match["bookmakers"] if b["key"] == BOOKMAKER_KEY), None
                    )
                    if not bookmaker:
                        continue

                    totals_market = next(
                        (m for m in bookmaker["markets"] if m["key"] == "totals"), None
                    )
                    if not totals_market:
                        continue

                    over_price = under_price = total_line = None

                    for outcome in totals_market["outcomes"]:
                        if outcome["name"] == "Over":
                            total_line = outcome["point"]
                            over_price = outcome["price"]
                        elif outcome["name"] == "Under":
                            under_price = outcome["price"]

                    if total_line is not None and over_price is not None and under_price is not None:
                        conn.execute(
                            text("""
                                UPDATE games
                                SET hr_total_runs_line = :line,
                                    over_price = :over,
                                    under_price = :under
                                WHERE id = :game_id
                            """),
                            {
                                "line": total_line,
                                "over": over_price,
                                "under": under_price,
                                "game_id": game_id,
                            }
                        )
                        print(f"✅ Updated game {game_id} ({away} @ {home}) with line {total_line}, Over {over_price}, Under {under_price}")
                        match_found = True
                        break

            if not match_found:
                print(f"❌ No line found for {away} @ {home} on {game_date}")

def update_games_with_lines():
    upcoming_games = fetch_upcoming_games()
    odds_data = fetch_totals_odds()

    with engine.begin() as conn:
        for game in upcoming_games:
            home = game["home_team"].lower()
            away = game["away_team"].lower()
            game_id = game["id"]

            match_found = None

            for match in odds_data:
                match_home = match["home_team"]
                match_away = match["away_team"]
                if team_names_match(home, match_home) and team_names_match(away, match_away):
                    bookmaker = next(
                        (b for b in match["bookmakers"] if b["key"] == "hardrockbet"), None
                    )
                    if not bookmaker:
                        continue

                    totals_market = next(
                        (m for m in bookmaker["markets"] if m["key"] == "totals"), None
                    )
                    if not totals_market:
                        continue

                    over_price = under_price = total_line = None

                    for outcome in totals_market["outcomes"]:
                        if outcome["name"] == "Over":
                            total_line = outcome["point"]
                            over_price = outcome["price"]
                        elif outcome["name"] == "Under":
                            under_price = outcome["price"]

                    if total_line is not None and over_price is not None and under_price is not None:
                        conn.execute(
                            text("""
                                UPDATE games
                                SET hr_total_runs_line = :line,
                                    over_price = :over,
                                    under_price = :under
                                WHERE id = :game_id
                            """),
                            {
                                "line": total_line,
                                "over": over_price,
                                "under": under_price,
                                "game_id": game_id,
                            }
                        )
                        print(f"✅ Updated game {game_id} with line {total_line}, Over {over_price}, Under {under_price}")
                        match_found = True
                        break

            if not match_found:
                print(f"❌ No line found for {away} @ {home}")


if __name__ == "__main__":
    update_games_with_lines()
    update_game_moneylines()