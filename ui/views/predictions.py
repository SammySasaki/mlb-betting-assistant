"""
Predictions view.

Renders the full predictions page: sidebar filters, summary metrics,
per-game cards, and a raw data table. Consumes the typed models from
`ui.api.client` — no raw HTTP calls live here.
"""

import datetime
from typing import Optional

import pandas as pd
import streamlit as st

from ui.api.client import get_games, get_predictions
from ui.api.models import Game, Prediction


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_odds(price: Optional[float]) -> str:
    if price is None:
        return "—"
    sign = "+" if price >= 0 else ""
    return f"{sign}{int(price)}"


def _fmt_prob(prob: Optional[float]) -> str:
    if prob is None:
        return "—"
    return f"{prob * 100:.1f}%"


def _fmt_edge(edge: Optional[float]) -> str:
    if edge is None:
        return "—"
    sign = "+" if edge >= 0 else ""
    return f"{sign}{edge:.2f}"


def _rec_color(recommendation: Optional[str]) -> str:
    if not recommendation or "NO BET" in recommendation.upper():
        return "gray"
    return "green"


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _load_games(date_str: str) -> list[Game]:
    return get_games(datetime.date.fromisoformat(date_str))


@st.cache_data(ttl=60)
def _load_predictions(date_str: str, exclude_no_bet: bool, market: Optional[str]) -> list[Prediction]:
    return get_predictions(
        datetime.date.fromisoformat(date_str),
        exclude_no_bet=exclude_no_bet,
        market=market or None,
    )


# ── Sub-components ────────────────────────────────────────────────────────────

def _render_odds_row(game: Game) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**O/U line:** {game.hr_total_runs_line or '—'}")
    c2.markdown(
        f"**Over:** {_fmt_odds(game.over_price)}  /  **Under:** {_fmt_odds(game.under_price)}"
    )
    c3.markdown(f"**{game.home_team} ML:** {_fmt_odds(game.home_ml_price)}")
    c4.markdown(f"**{game.away_team} ML:** {_fmt_odds(game.away_ml_price)}")


def _render_total_prediction(pred: Prediction, line: Optional[float]) -> None:
    c1, c2, c3, c4 = st.columns([2, 3, 2, 3])
    c1.markdown(f"**Market:** `{pred.market}`")
    total_str = f"{pred.predicted_total_runs:.2f}" if pred.predicted_total_runs is not None else "—"
    line_str = f"  *(line: {line})*" if line is not None else ""
    c2.markdown(f"**Predicted total:** {total_str}{line_str}")
    c3.markdown(f"**Edge:** {_fmt_edge(pred.edge)}")
    c4.markdown(
        f"<span style='color:{_rec_color(pred.recommendation)}; font-weight:bold; font-size:1.1em'>"
        f"{pred.recommendation or '—'}</span>",
        unsafe_allow_html=True,
    )


def _render_h2h_prediction(pred: Prediction, game: Game) -> None:
    c1, c2, c3, c4 = st.columns([2, 3, 2, 3])
    c1.markdown(f"**Market:** `{pred.market}`")
    c2.markdown(
        f"**{game.home_team}:** {_fmt_prob(pred.home_win_prob)}  |  "
        f"**{game.away_team}:** {_fmt_prob(pred.away_win_prob)}"
    )
    c3.markdown(f"**Edge:** {_fmt_edge(pred.edge)}")
    c4.markdown(
        f"<span style='color:{_rec_color(pred.recommendation)}; font-weight:bold; font-size:1.1em'>"
        f"{pred.recommendation or '—'}</span>",
        unsafe_allow_html=True,
    )


def _render_game_card(game: Game, preds: list[Prediction]) -> None:
    start_str = f"{game.start_hour_utc:02d}:00 UTC" if game.start_hour_utc is not None else "TBD"
    header = f"**{game.away_team}  @  {game.home_team}**  —  {start_str}  |  {game.venue or '—'}"

    with st.expander(header, expanded=True):
        _render_odds_row(game)
        st.markdown("---")
        for pred in sorted(preds, key=lambda p: p.market):
            if pred.market == "TOTAL":
                _render_total_prediction(pred, game.hr_total_runs_line)
            elif pred.market == "H2H":
                _render_h2h_prediction(pred, game)
            else:
                c1, c2, c3 = st.columns([2, 5, 3])
                c1.markdown(f"**Market:** `{pred.market}`")
                c2.markdown(f"**Edge:** {_fmt_edge(pred.edge)}")
                c3.markdown(
                    f"<span style='color:{_rec_color(pred.recommendation)}; font-weight:bold'>"
                    f"{pred.recommendation or '—'}</span>",
                    unsafe_allow_html=True,
                )


def _render_summary_metrics(games: list[Game], predictions: list[Prediction]) -> None:
    actionable = [
        p for p in predictions
        if p.recommendation and "NO BET" not in p.recommendation.upper()
    ]
    c1, c2, c3 = st.columns(3)
    c1.metric("Games today", len(games))
    c2.metric("Predictions", len(predictions))
    c3.metric("Actionable bets", len(actionable))


def _render_raw_data(games: list[Game], predictions: list[Prediction]) -> None:
    with st.expander("Raw prediction data"):
        if not predictions:
            st.write("No data.")
            return

        games_by_id = {g.id: g for g in games}
        rows = [
            {
                "game_id": p.game_id,
                "away_team": games_by_id[p.game_id].away_team if p.game_id in games_by_id else "",
                "home_team": games_by_id[p.game_id].home_team if p.game_id in games_by_id else "",
                "market": p.market,
                "recommendation": p.recommendation,
                "edge": p.edge,
                "predicted_total_runs": p.predicted_total_runs,
                "home_win_prob": p.home_win_prob,
                "away_win_prob": p.away_win_prob,
            }
            for p in predictions
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ── Sidebar controls ──────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[datetime.date, bool, Optional[str]]:
    """Renders sidebar filters and returns (selected_date, exclude_no_bet, market_filter)."""
    with st.sidebar:
        st.header("Filters")
        selected_date = st.date_input(
            "Date",
            value=datetime.date.today(),
            max_value=datetime.date.today() + datetime.timedelta(days=7),
        )
        exclude_no_bet = st.checkbox("Hide NO BET rows", value=False)
        market_raw = st.selectbox("Market", options=["All", "TOTAL", "H2H"], index=0)
        market_filter: Optional[str] = None if market_raw == "All" else market_raw

        st.divider()
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()

    return selected_date, exclude_no_bet, market_filter


# ── Public entry point ────────────────────────────────────────────────────────

def render() -> None:
    """Render the full predictions view. Called by home.py."""
    selected_date, exclude_no_bet, market_filter = _render_sidebar()

    try:
        games = _load_games(str(selected_date))
        predictions = _load_predictions(str(selected_date), exclude_no_bet, market_filter)
    except Exception as exc:  # noqa: BLE001
        if "ConnectionError" in type(exc).__name__:
            st.error("Cannot reach the API at **http://localhost:8000**. Make sure the FastAPI server is running.")
        else:
            st.error(f"API error: {exc}")
        return

    if not games:
        st.info(f"No games found for {selected_date}.")
        return

    _render_summary_metrics(games, predictions)
    st.divider()

    if not predictions:
        st.info("No predictions found for this date / filter combination.")
        _render_raw_data(games, predictions)
        return

    games_by_id = {g.id: g for g in games}
    preds_by_game: dict[int, list[Prediction]] = {}
    for pred in predictions:
        preds_by_game.setdefault(pred.game_id, []).append(pred)

    for game in games:
        game_preds = preds_by_game.get(game.id)
        if game_preds:
            _render_game_card(game, game_preds)

    _render_raw_data(games, predictions)
