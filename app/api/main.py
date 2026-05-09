import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import games, ingestion, odds, predictions, lineups, players, season_stats, chat

app = FastAPI(
    title="Sports Betting API",
    description="MLB game ingestion, odds management, lineup tracking, and bet predictions.",
    version="1.0.0",
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(games.router)
app.include_router(ingestion.router)
app.include_router(odds.router)
app.include_router(predictions.router)
app.include_router(lineups.router)
app.include_router(players.router)
app.include_router(season_stats.router)
app.include_router(chat.router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
