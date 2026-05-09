# mlb-betting-assistant

MLB game prediction and betting analysis tool with a LangGraph chatbot, FastAPI backend, and Streamlit UI.

## Project structure

```
app/
  agents/        LangGraph agent nodes (classifier, stats, lineup, predictor)
  api/           FastAPI app — routes, schemas, dependencies
  graph/         LangGraph workflow definition and state
  implementations/  Concrete implementations of interfaces (DB repos, API clients)
  ingestion/     One-off and scheduled data ingestion scripts
  interfaces/    Abstract interfaces for repos and API clients
  services/      Business logic
  utils/         Utility functions
infra/db/        SQLAlchemy engine setup and DB initialisation
ui/
  api/           HTTP client and typed models for the FastAPI backend
  views/         Streamlit page components (predictions, chat)
  home.py        Streamlit entry point
```

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ (or Docker)
- API keys for OpenAI, The Odds API, and Visual Crossing

## Quick start (Docker)

The easiest way to run everything is with Docker Compose, which starts Postgres, the FastAPI server, and the Streamlit UI together.

```bash
cp .env.example .env
# Fill in your API keys and review the DB credentials in .env
docker compose up --build
```

| Service    | URL                    |
|------------|------------------------|
| Streamlit UI | http://localhost:8501 |
| FastAPI    | http://localhost:8000  |
| PgAdmin    | http://localhost:5050  |

## Local development (without Docker)

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and set:

| Variable | Description |
|---|---|
| `DATABASE_URL` | SQLAlchemy connection string — use `localhost` for local Postgres, e.g. `postgresql://postgres:changeme@localhost:5432/sports_betting` |
| `OPENAI_API_KEY` | Required for the chatbot and prediction agents |
| `THE_ODDS_API_KEY` | Required for odds ingestion |
| `VISUAL_CROSSING_API_KEY` | Required for weather data |

### 4. Start Postgres

If you don't have Postgres running locally you can spin up just the DB container:

```bash
docker compose up db -d
```

### 5. Initialise the database

```bash
python infra/db/init_db.py
```

### 6. Run the FastAPI server

```bash
uvicorn app.api.main:app --reload
```

The API will be available at http://localhost:8000. Interactive docs are at http://localhost:8000/docs.

### 7. Run the Streamlit UI

In a separate terminal (with the virtual environment activated):

```bash
streamlit run ui/home.py
```

The UI will open at http://localhost:8501. It connects to the FastAPI server at `http://localhost:8000` by default. Override this with the `API_BASE_URL` environment variable if needed.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Send a message to the MLB chatbot |
| `GET` | `/predictions` | Retrieve stored predictions for a date |
| `POST` | `/predictions/run` | Run predictions for all games on a date |
| `GET` | `/games` | List games for a date |
| `GET` | `/odds` | Get odds lines |
| `GET` | `/lineups` | Get lineup data |
| `GET` | `/players` | Player lookup |
| `GET` | `/season-stats` | Player season stats |
| `GET` | `/health` | Health check |

## Data ingestion

After the server is running, use the ingestion routes or scripts to populate data before running predictions:

```bash
# Ingest upcoming games
python -m app.ingestion.ingest_upcoming

# Ingest odds
python -m app.ingestion.ingest_odds

# Run predictions for today
curl -X POST "http://localhost:8000/predictions/run?game_date=$(date +%F)"
```
