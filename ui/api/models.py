"""UI-side data models for the Sports Betting API responses."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Game:
    id: int
    home_team: str
    away_team: str
    venue: Optional[str]
    start_hour_utc: Optional[int]
    hr_total_runs_line: Optional[float]
    over_price: Optional[float]
    under_price: Optional[float]
    home_ml_price: Optional[float]
    away_ml_price: Optional[float]
    home_score: Optional[int]
    away_score: Optional[int]
    total_runs: Optional[float]

    @classmethod
    def from_dict(cls, data: dict) -> "Game":
        return cls(
            id=data["id"],
            home_team=data["home_team"],
            away_team=data["away_team"],
            venue=data.get("venue"),
            start_hour_utc=data.get("start_hour_utc"),
            hr_total_runs_line=data.get("hr_total_runs_line"),
            over_price=data.get("over_price"),
            under_price=data.get("under_price"),
            home_ml_price=data.get("home_ml_price"),
            away_ml_price=data.get("away_ml_price"),
            home_score=data.get("home_score"),
            away_score=data.get("away_score"),
            total_runs=data.get("total_runs"),
        )


@dataclass(frozen=True)
class Prediction:
    id: int
    game_id: int
    market: str
    predicted_total_runs: Optional[float]
    home_win_prob: Optional[float]
    away_win_prob: Optional[float]
    recommendation: Optional[str]
    edge: Optional[float]

    @classmethod
    def from_dict(cls, data: dict) -> "Prediction":
        return cls(
            id=data["id"],
            game_id=data["game_id"],
            market=data["market"],
            predicted_total_runs=data.get("predicted_total_runs"),
            home_win_prob=data.get("home_win_prob"),
            away_win_prob=data.get("away_win_prob"),
            recommendation=data.get("recommendation"),
            edge=data.get("edge"),
        )


@dataclass(frozen=True)
class NewsAlert:
    id: int
    source: str
    headline: str
    category: Optional[str]
    published_at: Optional[str]
    url: str
    teams: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "NewsAlert":
        return cls(
            id=data["id"],
            source=data["source"],
            headline=data["headline"],
            category=data.get("category"),
            published_at=data.get("published_at"),
            url=data.get("url", ""),
            teams=data.get("teams") or [],
        )
