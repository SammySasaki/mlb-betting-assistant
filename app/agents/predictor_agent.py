import joblib
import pandas as pd
import numpy as np
from infra.db.init_db import engine
from sqlalchemy import text, func, or_
from sqlalchemy.dialects import postgresql
from app.services.llm_service import call_llm
from app.services.feature_extractor import FeatureExtractor
from datetime import date, timedelta
import os
import json
from app.utils.utils import venue_orientations, venue_run_factors
from infra.db.init_db import SessionLocal
from sqlalchemy.orm import Session, joinedload
from app.db.models import Player, PlayerGameStats, Game, Prediction
from app.graph.state import GraphState
from app.utils.utils import llm_call_with_retry
from datetime import datetime
import pytz
from app.interfaces.illm_client import ILLMClient
from app.implementations.openai_client import OpenAILLMClient
from app.implementations.sqlalchemy_predictions_repository import SqlAlchemyPredictionRepository
from app.services.feature_extractor import FeatureExtractor


class MLBPredictionAgent:
    def __init__(self, openai_client: ILLMClient, db_session):
        self.openai_client = openai_client
        self.db = db_session
        self.prediction_repo = SqlAlchemyPredictionRepository(db_session)
        
    def _build_spec_prompt(self, user_message: str):
        return f"""
        You are a query translator that reads a natural language question and determines the recommendation that is being requested

        ### Context
        I have predictions for all of todays games for the total runs lines. The prediction info consists of the line, the over/under prices, the predicted total, 
        and a recommendation for each game.
        I also have predictions for the moneylines. Please indicate this market as H2H.
        
        ### Objective
        Convert the user's request into a **JSON spec** that includes info on what team the user is interested in, if specified. If both teams, specified only choose one.
        Also return the market that the user is interested in, if specified.
        If no team, or no market is specified, return 'ANY' for that field.

        ### Style
        - Output must be **valid JSON only** (no markdown, no extra text).


        ### Response Format
        {{
            "team": "<string>",
            "market": "<string>"
        }}

        ### Examples
        Request: "What should I bet for the Yankees game?"
        Response:
        {{
            "team": "New York Yankees",
            "market": "ANY"
        }}

        Request: "What are the best bets for total run lines for today?"
        Response:
        {{
            "team": "ANY",
            "market": "TOTAL"
        }}

        Request: "Should I bet over or under runs on the Giants vs Dodgers game today?"
        Response:
        {{
            "team": "San Francisco Giants",
            "market": "TOTAL"
        }}

        Request: "What are the best bets for moneylines today?"
        Response:
        {{
            "team": "ANY",
            "market": "H2H"
        }}

        Request: "Who should I bet on for the red sox game tonight?"
        Response:
        {{
            "team": "Boston Red Sox",
            "market": "H2H"
        }}

        ### Task
        Now generate the JSON spec for:
        "{user_message}"
        """
    
    def spec_to_rec(self, spec: dict):
            # Get today in Eastern timezone
            eastern_tz = pytz.timezone("America/New_York")
            ny_dt = datetime.now(eastern_tz)
            today = ny_dt.date()

            team = spec.get("team", "ANY")
            market = spec.get("market", "ANY")

            # Call repo
            predictions = self.prediction_repo.get_predictions(
                game_date=today,
                team=team,
                market=market,
                exclude_no_bet=(team == "ANY")
            )

            # Format results
            recs = []
            for pred in predictions:
                rec = {
                    "game_id": pred.game_id,
                    "home_team": pred.game.home_team,
                    "away_team": pred.game.away_team,
                    "recommendation": pred.recommendation,
                    "market": pred.market,
                }

                if pred.market == "TOTAL":
                    rec["predicted_total_runs"] = pred.predicted_total_runs
                    rec["edge"] = pred.edge
                elif pred.market == "H2H":
                    rec["home_win_prob"] = pred.home_win_prob
                    rec["away_win_prob"] = pred.away_win_prob

                recs.append(rec)

            return recs

    def recs_to_str(self, recs):
        output = ""
        for r in recs:
            home = r["home_team"]
            away = r["away_team"]
            market = r["market"]
            output += f"[{market}] {away} @ {home}\n"
            if market == "TOTAL":
                output += f"  Predicted Total Runs: {r['predicted_total_runs']:.2f}\n"
            elif market == "H2H":
                output += f"  Home ({home}) Win Prob: {r['home_win_prob']:.2f}\n"
                output += f"  Away ({away}) Win Prob: {r['away_win_prob']:.2f}\n"
            rec_str = r["recommendation"]
            rec_str = rec_str.replace("HOME", f"HOME ({home})").replace("AWAY", f"AWAY ({away})")
            output += f"  Recommendation: {rec_str}\n"
            if r.get("edge") is not None:
                output += f"  Edge: {r['edge']:.2f}\n"
            output += "\n"
        return output


    def fetch_predictions(self, state: GraphState) -> GraphState:
        """RICH_RECOMMENDATION path: fetch structured predictions into state, no output string."""
        user_message = state["input"]
        spec_prompt = self._build_spec_prompt(user_message)
        spec = llm_call_with_retry(
            self.openai_client,
            messages=[{"role": "system", "content": spec_prompt}],
        )
        recs = self.spec_to_rec(spec)
        return {**state, "predictions_data": recs}

    def _build_context_block(self, preds: list, context: dict) -> str:
        lines = []
        for rec in preds:
            gid = str(rec["game_id"])
            home = rec["home_team"]
            away = rec["away_team"]
            lines.append(f"## {away} @ {home}  (HOME={home}, AWAY={away})")
            rec_str = rec["recommendation"]
            rec_str = rec_str.replace("HOME", f"HOME ({home})").replace("AWAY", f"AWAY ({away})")
            lines.append(f"Market: {rec['market']}, Recommendation: {rec_str}")
            if rec.get("predicted_total_runs") is not None:
                lines.append(f"Predicted Total: {rec['predicted_total_runs']:.2f}, Edge: {rec.get('edge', 0) or 0:.3f}")
            if rec.get("home_win_prob") is not None:
                lines.append(f"Home Win Prob: {rec['home_win_prob']:.2f}, Away Win Prob: {rec['away_win_prob']:.2f}")
            ctx = context.get(gid, {})
            for side in ("home_team_stats", "away_team_stats"):
                stats = ctx.get(side)
                if stats:
                    s = stats[0]
                    lines.append(
                        f"  {s['team']} last {s['last_n_games']}g: "
                        f"AVG {s['batting_avg']}, {s['home_runs']} HR, {s['runs']} R"
                    )
            news = ctx.get("news", [])
            if news:
                lines.append("  News:")
                for n in news:
                    lines.append(f"    - {n}")
            lines.append("")
        return "\n".join(lines)

    def synthesize_recommendation(self, state: GraphState) -> GraphState:
        """RICH_RECOMMENDATION path: combine predictions + context into a narrative response."""
        user_message = state["input"]
        preds = state.get("predictions_data") or []
        context = state.get("context_data") or {}

        if not preds:
            return {**state, "output": "No predictions found for today's games matching your request."}

        context_block = self._build_context_block(preds, context)
        synthesis_prompt = (
            f'The user asked: "{user_message}"\n\n'
            "Below are today's model predictions and supporting context. "
            "Write a thorough, well-reasoned recommendation: explain the key factors "
            "supporting each bet (model edge, recent offensive/pitching trends, relevant news). "
            "Be direct about confidence. One section per game.\n\n"
            f"{context_block}"
        )
        response = self.openai_client.chat(
            messages=[
                {"role": "system", "content": "You are an expert MLB betting analyst. Be analytical and specific."},
                {"role": "user", "content": synthesis_prompt},
            ],
        )
        return {**state, "output": response}

    def handle_request(self, state: GraphState):
        """RECOMMENDATION path: natural language request -> spec -> DB -> formatted string."""
        user_message = state["input"]
        spec_prompt = self._build_spec_prompt(user_message)
        spec = llm_call_with_retry(
            self.openai_client,
            messages=[{"role": "system", "content": spec_prompt}],
        )
        recs = self.spec_to_rec(spec)
        output = self.recs_to_str(recs)
        return {**state, "output": output}

if __name__ == "__main__":
    openai_client = OpenAILLMClient(api_key=os.getenv("OPENAI_API_KEY"))
    db = SessionLocal()
    agent = MLBPredictionAgent(openai_client=openai_client, db_session=db)
    state = GraphState(input="What should I bet today for the Giants game?", intent="RECOMMENDATION")
    state = GraphState(input="What should I bet today?", intent="RECOMMENDATION")
    res = agent.handle_request(state)
    print(res['output'])