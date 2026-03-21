"""Simple disk-backed cache for external API responses.

Historical data (any date before today) is stored permanently and never
re-fetched.  Data for today is stored with a configurable TTL so it is
refreshed periodically without burning quota on every run.
"""

import json
import os
import time
from datetime import date

_CACHE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", ".cache")


def _path(namespace: str, key: str) -> str:
    directory = os.path.join(_CACHE_ROOT, namespace)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"{key}.json")


def load(namespace: str, key: str, target_date: date | None = None, ttl: int = 7200) -> dict | list | None:
    """Return cached data or None if absent / stale.

    Args:
        namespace: Subdirectory name (e.g. "weather", "odds").
        key:       Cache filename without extension.
        target_date: When provided, historical dates (before today) never expire.
        ttl:       Seconds before today's data is considered stale (default 2 h).
    """
    filepath = _path(namespace, key)
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        entry = json.load(f)

    # Historical data never expires.
    if target_date is not None and target_date < date.today():
        return entry["data"]

    if time.time() - entry["timestamp"] < ttl:
        return entry["data"]

    return None


def save(namespace: str, key: str, data: dict | list) -> None:
    """Persist data to the disk cache."""
    filepath = _path(namespace, key)
    with open(filepath, "w") as f:
        json.dump({"timestamp": time.time(), "data": data}, f)
