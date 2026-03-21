from pydantic import BaseModel
from datetime import date
from typing import Optional, List


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

class GameResponse(BaseModel):
    id: int
    date: date
    season_year: Optional[int] = None
    home_team: str
    away_team: str
    venue: Optional[str] = None
    start_hour_utc: Optional[int] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    total_runs: Optional[float] = None
    hr_total_runs_line: Optional[float] = None
    over_price: Optional[float] = None
    under_price: Optional[float] = None
    home_ml_price: Optional[float] = None
    away_ml_price: Optional[float] = None
    home_probable_pitcher_id: Optional[int] = None
    away_probable_pitcher_id: Optional[int] = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

class IngestDayRequest(BaseModel):
    date: date


class IngestRangeRequest(BaseModel):
    start_date: date
    end_date: date


# ---------------------------------------------------------------------------
# Odds
# ---------------------------------------------------------------------------

class OddsDateRequest(BaseModel):
    date: date


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

class PredictionResponse(BaseModel):
    id: int
    game_id: int
    market: str
    predicted_total_runs: Optional[float] = None
    home_win_prob: Optional[float] = None
    away_win_prob: Optional[float] = None
    recommendation: Optional[str] = None
    edge: Optional[float] = None

    model_config = {"from_attributes": True}


class PredictionsQueryParams(BaseModel):
    date: date
    team: Optional[str] = None
    market: Optional[str] = None
    exclude_no_bet: bool = False


# ---------------------------------------------------------------------------
# Lineups
# ---------------------------------------------------------------------------

class LineupDateRequest(BaseModel):
    date: date


# ---------------------------------------------------------------------------
# Players / Matchup
# ---------------------------------------------------------------------------

class MatchupQueryParams(BaseModel):
    opponent_id: int
    position: str
    season: Optional[int] = None
    limit: Optional[int] = None
    select_fields: Optional[List[str]] = None
    aggregate: Optional[str] = None


# ---------------------------------------------------------------------------
# Season Stats
# ---------------------------------------------------------------------------

class PlayerSeasonStatsResponse(BaseModel):
    player_id: int
    season_year: int

    # Batting counting
    games: Optional[int] = None
    at_bats: Optional[int] = None
    hits: Optional[int] = None
    doubles: Optional[int] = None
    triples: Optional[int] = None
    home_runs: Optional[int] = None
    rbis: Optional[int] = None
    runs: Optional[int] = None
    walks_batting: Optional[int] = None

    # Batting rates
    avg: Optional[float] = None
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None

    # Pitching counting
    games_pitched: Optional[int] = None
    outs_pitched: Optional[int] = None
    earned_runs: Optional[int] = None
    strikeouts: Optional[int] = None
    walks_pitched: Optional[int] = None
    hits_allowed: Optional[int] = None

    # Pitching rates
    era: Optional[float] = None
    whip: Optional[float] = None

    model_config = {"from_attributes": True}
