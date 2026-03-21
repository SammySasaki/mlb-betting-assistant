from fastapi import FastAPI

from app.api.routes import games, ingestion, odds, predictions, lineups, players, season_stats

app = FastAPI(
    title="Sports Betting API",
    description="MLB game ingestion, odds management, lineup tracking, and bet predictions.",
    version="1.0.0",
)

app.include_router(games.router)
app.include_router(ingestion.router)
app.include_router(odds.router)
app.include_router(predictions.router)
app.include_router(lineups.router)
app.include_router(players.router)
app.include_router(season_stats.router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
