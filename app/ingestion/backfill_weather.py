from zoneinfo import ZoneInfo
import requests
from datetime import datetime
from app.db_models import Game, Weather
from sqlalchemy.orm import Session
from infra.db.init_db import SessionLocal
from app.utils.utils import venue_utc_offsets

VISUAL_CROSSING_KEY = "WH38HYPTPCX88DGXD3ERUEQY8"
session: Session = SessionLocal()

def fetch_hourly_weather(city, date_str, target_hour):
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{date_str}"
    params = {
        "unitGroup": "us",
        "key": VISUAL_CROSSING_KEY,
        "include": "hours",
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching weather for {city} on {date_str}: {response.status_code}")
        print("Response text:", response.text)
        if response.text == 'Maximum daily cost exceeded':
            return "out of credits"
        return None
    data = response.json()
    if "days" not in data:
        return None

    # Find the hour closest to the game's start_time
    hours = data["days"][0]["hours"]
    for hour_data in hours:
        hour = datetime.strptime(hour_data["datetime"], "%H:%M:%S").hour
        if hour == target_hour:
            return {
                "temperature": hour_data.get("temp"),
                "wind_speed": hour_data.get("windspeed"),
                "wind_direction": hour_data.get("winddir"),
                "precipitation": hour_data.get("precip"),
            }

    return None  # fallback if exact hour not found

def map_venue_to_city(venue_name):
    mappings = {
        "Fenway Park": "Boston, MA",
        "Yankee Stadium": "New York, NY",
        "Dodger Stadium": "Los Angeles, CA",
        "Wrigley Field": "Chicago, IL",
        "Truist Park": "Atlanta, GA",
        "Oriole Park at Camden Yards": "Baltimore, MD",
        "George M. Steinbrenner Field": "Tampa, FL",
        "Rogers Centre": "Toronto, ON",
        "Comerica Park": "Detroit, MI",
        "Kauffman Stadium": "Kansas City, MO",
        "Angel Stadium": "Anaheim, CA",
        "Daikin Park": "Houston, TX",
        "Target Field": "Minneapolis, MN",
        "Sutter Health Park": "Oakland, CA",
        "T-Mobile Park": "Seattle, WA",
        "Progressive Field": "Cleveland, OH",
        "Globe Life Field": "Arlington, TX",
        "Rate Field": "Chicago, IL",
        "PNC Park": "Pittsburgh, PA",
        "Citizens Bank Park": "Philadelphia, PA",
        "Petco Park": "San Diego, CA",
        "Oracle Park": "San Francisco, CA",
        "American Family Field": "Milwaukee, WI",
        "Busch Stadium": "St. Louis, MO",
        "Nationals Park": "Washington, DC",
        "Great American Ball Park": "Cincinnati, OH",
        "loanDepot park": "Miami, FL",
        "Chase Field": "Phoenix, AZ",
        "Coors Field": "Denver, CO",
        "Citi Field": "New York, NY"
    }
    return mappings.get(venue_name)

def get_local_hour(start_hour_utc: int, venue: str) -> int:
    offset = venue_utc_offsets.get(venue)
    if offset is None:
        raise ValueError(f"No UTC offset defined for venue: {venue}")
    
    # Convert to local hour with 24-hour wraparound
    local_hour = (start_hour_utc + offset) % 24
    return local_hour

def backfill_weather():
    games = (
        session.query(Game)
        .order_by(Game.date.desc())
        .all()
    )
    for game in games:
        city = map_venue_to_city(game.venue)
        if not city:
            continue

        existing_weather = session.query(Weather).filter_by(game_id=game.id).first()

        if not existing_weather:

            local_hour = get_local_hour(game.start_hour_utc, game.venue)
            weather_data = fetch_hourly_weather(city, game.date.isoformat(), local_hour)
            if isinstance(weather_data, str) and "out of credits" in weather_data.lower():
                print("Out of credits — committing progress and stopping.")
                session.commit()
                break

            if not weather_data:
                continue


            
            # Create new weather
            weather = Weather(
                game_id=game.id,
                temperature=weather_data["temperature"],
                wind_speed=weather_data["wind_speed"],
                wind_direction=str(weather_data["wind_direction"]),
                precipitation=weather_data["precipitation"] > 0
            )
            session.add(weather)
            print(f"Added weather for {game.venue} on {game.date}")
            session.commit()

if __name__ == "__main__":
    backfill_weather()