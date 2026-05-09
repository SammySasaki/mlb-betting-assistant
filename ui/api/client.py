"""
HTTP client for the Sports Betting FastAPI backend.

All network calls go through this module. Each function raises
`requests.HTTPError` on a non-2xx response and
`requests.ConnectionError` when the server is unreachable.
"""

import datetime
from typing import Optional

import os

import requests

from ui.api.models import Game, Prediction

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def get_games(game_date: datetime.date) -> list[Game]:
    resp = requests.get(
        f"{API_BASE}/games",
        params={"date": str(game_date)},
        timeout=10,
    )
    resp.raise_for_status()
    return [Game.from_dict(g) for g in resp.json()]


def send_chat_message(message: str) -> str:
    resp = requests.post(
        f"{API_BASE}/chat",
        json={"message": message},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def get_predictions(
    game_date: datetime.date,
    *,
    exclude_no_bet: bool = False,
    market: Optional[str] = None,
) -> list[Prediction]:
    params: dict = {
        "game_date": str(game_date),
        "exclude_no_bet": str(exclude_no_bet).lower(),
    }
    if market:
        params["market"] = market

    resp = requests.get(
        f"{API_BASE}/predictions",
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return [Prediction.from_dict(p) for p in resp.json()]
