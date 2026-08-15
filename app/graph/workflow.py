# graph/workflow.py
from langgraph.graph import StateGraph, END, START
from app.agents.classifier_agent import ClassifierAgent
from app.agents.stats_agent import StatsAgent
from app.agents.live_stat_agent import LiveStatAgent
from app.agents.predictor_agent import MLBPredictionAgent
from app.agents.lineup_agent import LineupAgent
from app.agents.news_agent import NewsAgent
from app.agents.context_agent import ContextAgent
import os
from openai import OpenAI
from app.implementations.openai_client import OpenAILLMClient
from infra.db.init_db import SessionLocal
from app.graph.state import GraphState
from app.services.at_bat_service import AtBatService
from app.implementations.sqlalchemy_player_repository import SQLAlchemyPlayerRepository
from app.implementations.sqlalchemy_lineup_repository import SQLAlchemyLineupRepository
from app.implementations.sqlalchemy_pgs_repository import SqlAlchemyPlayerGameStatsRepository
from app.implementations.sqlalchemy_game_repository import GameRepository
from app.implementations.mlb_api_client import StatsApiClient
from app.services.lineup_service import LineupService

def classifier_router(state: GraphState) -> str:
    intent = state["intent"]

    if intent == "STAT":
        return "STAT"
    elif intent == "LIVE_STAT":
        return "LIVE_STAT"
    elif intent == "RECOMMENDATION":
        return "RECOMMENDATION"
    elif intent == "RICH_RECOMMENDATION":
        return "RICH_RECOMMENDATION"
    elif intent == "LINEUP":
        return "LINEUP"
    elif intent == "NEWS":
        return "NEWS"
    elif intent == "OBSERVATION":
        return "OBSERVATION"
    else:
        return "END"

def build_graph():
    openai_client = OpenAILLMClient(api_key=os.getenv("OPENAI_API_KEY"))
    db = SessionLocal()
    atbat_service = AtBatService()
    player_repository = SQLAlchemyPlayerRepository(db)
    lineup_repository = SQLAlchemyLineupRepository(db)
    pgs_repository = SqlAlchemyPlayerGameStatsRepository(db)
    game_repository = GameRepository(db)
    mlb_api_client = StatsApiClient()
    lineup_service = LineupService(pgs_repository, lineup_repository, mlb_api_client, game_repository, player_repository)

    def observation_handler(state: GraphState) -> GraphState:
        reply = openai_client.chat([
            {"role": "system", "content": (
                "You are a sports betting assistant. The user made a casual observation about a team or player. "
                "Acknowledge it briefly in one or two sentences, then offer to help — for example, suggest you can "
                "pull up recent stats, check today's lineup, or give a bet recommendation."
            )},
            {"role": "user", "content": state["input"]},
        ])
        return {**state, "output": reply}

    classifier = ClassifierAgent(openai_client)
    stat_agent = StatsAgent(db, openai_client, atbat_service)
    live_stat_agent = LiveStatAgent(mlb_api_client, openai_client, player_repository)
    lineup_agent = LineupAgent(player_repository, lineup_repository, openai_client, lineup_service, game_repository)
    prediction_agent = MLBPredictionAgent(openai_client=openai_client, db_session=db)
    news_agent = NewsAgent(openai_client=openai_client, db_session=db)
    context_agent = ContextAgent(db_session=db, n_recent_games=5)

    graph = StateGraph(GraphState)

    # --- Nodes ---
    graph.add_node("classifier", classifier.classify_message)
    graph.add_node("observation_agent", observation_handler)

    graph.add_node("stat_agent", stat_agent.handle_request)
    graph.add_node("live_stat_agent", live_stat_agent.handle_request)
    graph.add_node("prediction_agent", prediction_agent.handle_request)
    graph.add_node("fetch_predictions", prediction_agent.fetch_predictions)
    graph.add_node("context_agent", context_agent.gather_context)
    graph.add_node("synthesize_recommendation", prediction_agent.synthesize_recommendation)
    graph.add_node("lineup_agent", lineup_agent.handle_request)
    graph.add_node("news_agent", news_agent.handle_request)

    # --- Flow ---
    # 1. Classifier runs → produces state with `intent`
    graph.add_edge(START, "classifier")
    graph.add_conditional_edges(
        "classifier",
        classifier_router,
        {
            "STAT": "stat_agent",
            "LIVE_STAT": "live_stat_agent",
            "RECOMMENDATION": "prediction_agent",
            "RICH_RECOMMENDATION": "fetch_predictions",
            "LINEUP": "lineup_agent",
            "NEWS": "news_agent",
            "OBSERVATION": "observation_agent",
            "END": END
        },
    )

    graph.add_edge("stat_agent", END)
    graph.add_edge("live_stat_agent", END)
    graph.add_edge("prediction_agent", END)
    graph.add_edge("lineup_agent", END)
    graph.add_edge("news_agent", END)
    graph.add_edge("observation_agent", END)

    graph.add_edge("fetch_predictions", "context_agent")
    graph.add_edge("context_agent", "synthesize_recommendation")
    graph.add_edge("synthesize_recommendation", END)

    return graph.compile()