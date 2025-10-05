from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, ForeignKey, Time, Boolean, func, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Game(Base):
    __tablename__ = "games"

    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False)
    season_year = Column(Integer)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    venue = Column(String)
    start_hour_utc = Column(Integer)
    total_runs = Column(Float)  # sum of both teams’ final score
    home_score = Column(Integer)
    away_score = Column(Integer)

    hr_total_runs_line = Column(Float)
    over_price = Column(Float)
    under_price = Column(Float)

    home_ml_price = Column(Float)
    away_ml_price = Column(Float)

    home_probable_pitcher_id = Column(Integer, ForeignKey("players.id"), nullable=True)
    away_probable_pitcher_id = Column(Integer, ForeignKey("players.id"), nullable=True)

    # Relationships
    weather = relationship("Weather", back_populates="game", uselist=False, cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="game", cascade="all, delete-orphan")
    player_stats = relationship("PlayerGameStats", back_populates="game", cascade="all, delete-orphan")
    home_probable_pitcher = relationship("Player", foreign_keys=[home_probable_pitcher_id])
    away_probable_pitcher = relationship("Player", foreign_keys=[away_probable_pitcher_id])
    lineup_entries = relationship("Lineups", back_populates="game", cascade="all, delete-orphan")
    team_features = relationship("TeamFeatures", back_populates="game")
    pitcher_features = relationship("PitcherFeatures", back_populates="game")
    bullpen_features = relationship("BullpenFeatures", back_populates="game")

class Weather(Base):
    __tablename__ = "weather"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), unique=True)
    temperature = Column(Float)
    wind_speed = Column(Float)
    wind_direction = Column(String)
    precipitation = Column(Boolean)
    notes = Column(String)

    game = relationship("Game", back_populates="weather")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    team = Column(String)
    position = Column(String)
    throwing_hand = Column(String)

    game_stats = relationship("PlayerGameStats", back_populates="player")
    pitcher_features = relationship("PitcherFeatures", back_populates="player")
    lineup_entries = relationship("Lineups", back_populates="player")

class PlayerGameStats(Base):
    __tablename__ = "player_game_stats"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    player_id = Column(Integer, ForeignKey("players.id"))
    team = Column(String)

    # Batting
    at_bats = Column(Integer)
    hits = Column(Integer)
    runs = Column(Integer)
    home_runs = Column(Integer)
    rbis = Column(Integer)
    walks_batting = Column(Integer)
    doubles = Column(Integer)
    triples = Column(Integer)

    # Pitching
    innings_pitched = Column(Float)
    outs_pitched = Column(Integer)
    earned_runs = Column(Integer)
    strikeouts = Column(Integer)
    walks = Column(Integer)
    hits_allowed = Column(Integer)

    game = relationship("Game", back_populates="player_stats")
    player = relationship("Player", back_populates="game_stats")


class TeamFeatures(Base):
    __tablename__ = "team_features"

    team = Column(String, primary_key=True)  # team code, e.g. "NYY"
    game_id = Column(Integer, ForeignKey("games.id"), primary_key=True)
    season_year = Column(Integer, nullable=False)

    wins_season = Column(Integer)
    losses_season = Column(Integer)

    wins_last10 = Column(Integer)
    avg_runs_last10 = Column(Float)
    run_diff_season = Column(Integer)

    created_at = Column(DateTime, server_default=func.now())

    game = relationship("Game", back_populates="team_features")


# --- PITCHER FEATURES ---
class PitcherFeatures(Base):
    __tablename__ = "pitcher_features"

    player_id = Column(Integer, ForeignKey("players.id"), primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), primary_key=True)
    season_year = Column(Integer, nullable=False)

    sp_era = Column(Float)      # season-to-date ERA before game
    sp_whip = Column(Float)     # WHIP before game
    last3_era = Column(Float)   # ERA over last 3 starts

    created_at = Column(DateTime, server_default=func.now())

    game = relationship("Game", back_populates="pitcher_features")
    player = relationship("Player", back_populates="pitcher_features")


# --- BULLPEN FEATURES ---
class BullpenFeatures(Base):
    __tablename__ = "bullpen_features"

    team = Column(String, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"), primary_key=True)
    season_year = Column(Integer, nullable=False)

    bullpen_era = Column(Float)

    created_at = Column(DateTime, server_default=func.now())

    game = relationship("Game", back_populates="bullpen_features")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    market = Column(String, nullable=False)  
    predicted_total_runs = Column(Float)
    home_win_prob = Column(Float)
    away_win_prob = Column(Float)
    recommendation = Column(String)
    edge = Column(Float)

    game = relationship("Game", back_populates="predictions")

class Lineups(Base):
    __tablename__ = "lineup_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)

    game_id = Column(Integer, ForeignKey("games.id", ondelete="CASCADE"), nullable=False)
    batting_order = Column(Integer, nullable=False)
    player_id = Column(Integer, ForeignKey("players.id"), nullable=False)

    defensive_position = Column(String(3), nullable=True)

    # Pregame batter stats
    avg_season = Column(Float)
    obp_season = Column(Float)
    slg_season = Column(Float)
    ops_season = Column(Float)
    home_runs = Column(Integer)
    rbis = Column(Integer) 
    recent_ops = Column(Float)

    # Relationships
    game = relationship("Game", back_populates="lineup_entries")
    player = relationship("Player", back_populates="lineup_entries")

    __table_args__ = (
        UniqueConstraint("game_id", "batting_order", name="uq_lineup_slot"),
    )


class UserEvent(Base):
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True)
    game_id = Column(Integer, ForeignKey("games.id"))
    event_type = Column(String)  # e.g. "injury", "weather", "lineup_change"
    description = Column(String)  # freeform, maybe from LLM
    affected_team = Column(String)
    affected_players = Column(String)  # comma-separated or JSON
    created_by = Column(String)  # "user", "LLM", etc.
    created_at = Column(DateTime)