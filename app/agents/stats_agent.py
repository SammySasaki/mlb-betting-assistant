from sqlalchemy.orm import Session, aliased
from sqlalchemy import desc, asc, case, and_, or_, func
from sqlalchemy.dialects import postgresql
from app.db_models import Game, Player, PlayerGameStats, Weather
from app.services.at_bat_service import AtBatService
import os
from openai import OpenAI
import json, re
from infra.db.init_db import SessionLocal
from app.graph.state import GraphState
from app.utils.utils import _extract_json
from app.interfaces.illm_client import ILLMClient
from app.implementations.openai_client import OpenAILLMClient

openai_client = OpenAILLMClient(api_key=os.getenv("OPENAI_API_KEY"))
db = SessionLocal()



class StatsAgent:
    def __init__(self, db_session, openai_client: ILLMClient, atbat_service, model="gpt-4o-mini"):
        self.openai_client = openai_client
        self.db_session = db_session
        self.model = model
        self.atbat_service = atbat_service

    def _build_spec_prompt(self, user_message: str):
        return f"""
        You are a query translator that converts natural language MLB questions 
        into a structured JSON spec for querying my SQLAlchemy tables.

        ### Context
        I have MLB game and player data stored in PostgreSQL with SQLAlchemy models.
        The relevant tables are:

        - Game:
            id, date, season_year, home_team, away_team, venue,
            start_hour_utc, total_runs, home_score, away_score,
            hr_total_runs_line, over_price, under_price,
            home_probable_pitcher_id, away_probable_pitcher_id

        - Weather:
            game_id, temperature, wind_speed, wind_direction, precipitation, notes

        - Player:
            id, name, team, position, throwing_hand

        - PlayerGameStats:
            game_id, player_id, team,
            at_bats, hits, runs, home_runs, rbis, walks_batting, doubles, triples,
            outs_pitched, earned_runs, strikeouts, walks, hits_allowed

        - At-bat matchup service (external):
            Provides batter vs. pitcher results (e.g., hits, at-bats, strikeouts) for a given season


        The current season_year is 2025

        ### Objective
        Convert the user's request into a **JSON spec** that fully supports:
        1. Selecting stats from any table (`Game`, `Weather`, `Player`, `PlayerGameStats`).
        2. Filtering on any field in any table.
        3. Joining across tables when necessary (e.g., PlayerGameStats → Player → Game).
        4. Aggregating across multiple rows when implied (e.g., total or sum of `home_runs`).
        5. If the user asks for a **batter vs. pitcher matchup**, output `at_bat_filters` so the request can be redirected to the at-bat service.
        6. For any player name or team given, please fix any spelling errors or formatting errors.

        ### Style
        - Output must be **valid JSON only** (no markdown, no extra text).
        - Use field names exactly as defined in the schema.
        - Only include relevant filters.
        - Include `"select"` when the user asks for a specific stat.
        - Include `"aggregate"` when the user is asking for a total, average, or sum.

        ### Select Vocabulary
        `select` may be a string or array of strings. Supported values include:

        - Game fields: "date", "home_team", "away_team", "venue", "total_runs", "home_score", "away_score", "hr_total_runs_line", "over_price", "under_price"
        - Weather fields: "temperature", "wind_speed", "wind_direction", "precipitation"
        - Player fields: "name", "team", "position", "throwing_hand"
        - PlayerGameStats fields: "at_bats", "hits", "runs", "home_runs", "rbis", "walks_batting", "doubles", "triples", "outs_pitched", "earned_runs", "strikeouts", "walks", "hits_allowed"
        - At-bat service fields: "at_bats", "hits", "strikeouts", "home_runs", "batting_average", "events" etc.
            - an "event" is the result of the at-bat, so like single, strikeout, walk, etc
        - Computed aliases:
            - "runs_scored" → returns `home_score` if `home_team == filters.team` else `away_score`
            - "opponent_runs" → returns `away_score` if `home_team == filters.team` else `home_score`
        
        populate selects and filters ONLY with column names or computed aliases


        ### Filters
        Filters may include fields from any table:
        - `game_filters`: any fields from Game
        - `weather_filters`: any fields from Weather
        - `player_filters`: any fields from Player
        - `player_stats_filters`: any fields from PlayerGameStats
        - `at_bat_filters`: opposing player field `{{"opposing_player": "Aaron Judge"}}` and player_type field `{{"player_type": "P"}}
            - player_type is the type(Pitcher P or Batter B) of the player user wants stats for
        - Example: `player_filters.name`, `player_stats_filters.home_runs`, `game_filters.season_year`, `weather_filters.temperature_gt`

        ### Aggregation
        If the user asks for totals, averages, or sums, include:
        ```json
        "aggregate": "sum" | "avg" | "count"

        ### Calculations
        If the user asks for a calculated stat (e.g., AVG, ERA), include the necessary fields in `select`,
         specify the calculation, and leave the calculation to be done outside SQL.

        ### Response Format
        {{
            "select": "<string or string[]>",
            "filters": {{
                "game_filters": {{ ... }},
                "weather_filters": {{ ... }},
                "player_filters": {{ ... }},
                "player_stats_filters": {{ ... }}
            }},
            "aggregate": "<optional: sum|avg|count>",
            "order_by": "date_asc" | "date_desc",
            "calculate": "<optional: description of calculation>",
            "limit": <int>
        }}

        ### Examples
        Request: "Show me the last 5 Yankee games in hot weather"
        Response:
        {{
            "select": ["date", "home_team", "away_team", "total_runs"],
            "filters": {{
                "game_filters": {{"home_team": "New York Yankees"}},
                "weather_filters": {{"temperature_gt": 80}}
            }},
            "order_by": "date_desc",
            "limit": 5
        }}

        Request: "How many home runs did Shohei Ohtani have this season?"
        Response:
        {{
        "select": "home_runs",
        "filters": {{
            "player_filters": {{"name": "Shohei Ohtani"}},
            "game_filters": {{"season_year": 2025}}
        }},
        "aggregate": "sum"
        }}

        Request: "What is Shohei's batting average this year?"
        Response:
        {{
        "select": ["hits", "at_bats"], 
        "filters": {{
            "game_filters": {{"season_year": 2025}}, 
            "player_filters": {{"name": "Shohei Ohtani"}}, 
        }},
        "aggregate": "sum",
        "calculate": "AVG"
        }}

        Request: "How many wins do the Giants have this season?"
        Response
        {{
        "select": "runs_scored",
        "filters": {{
            "game_filters": {{"season_year": 2025, "team": "San Francisco Giants", "runs_scored_gt": opponent_runs}},
        }},
        "aggregate": "count"
        }}

        Request: "How many runs did the Yankees score in each of their last 5 games?"
        Response:
        {{
        "select": "runs_scored",
        "filters": {{
            "game_filters": {{"team": "New York Yankees"}}
        }},
        "order_by": "date_desc",
        "limit": 5
        }}

        Request: "How many total runs have the Giants scored in the last 3 games against the dodgers"
        Response:
        {{
        "select": "runs_scored",
        "filters": {{
            "game_filters": {{"team": "San Francisco Giants", "opponent_team": "Los Angeles Dodgers"}}
        }},
        "order_by": "date_desc",
        "limit": 3,
        "aggregate": "sum"
        }}

        Request: "How many hits does Shohei Ohtani have against Tarik Skubal this season?"
        Response:
        {{
        "select": "hits",
        "filters": {{
            "game_filters": {{"season_year": 2025}},
            "player_filters": {{"name": "Shohei Ohtani"}},
            "at_bat_filters": {{ "opposing_player": "Tarik Skubal", "player_type": "B" }}
        }},
        "aggregate": "sum"
        }}

        Request: "How did Kevin Gausman do in his last 5 at bats against Aaron Judge?"
        {{
        "select": "events",
        "filters": {{
            "player_filters": {{"name": "Kevin Gausman"}},
            "at_bat_filters": {{ "opposing_player": "Aaron Judge", "player_type": "P" }}
        }},
        }}

        Request: "What is Cam Schlittler's ERA?"
        {{
        "select": ["earned_runs", "outs_pitched"],
        "filters": {{
            "game_filters": {{"season_year": 2025}},
            "player_filters": {{"name": "Cam Schlittler"}}
        }},
        "aggregate": "sum",
        "calculate": "ERA"
        }}
        ### Task 
        Now generate the JSON spec for: "{user_message}"
        """
    
    def _build_answer_prompt(self, user_message: str, spec: dict, results: list) -> str:
        # Give the LLM the results as context and ask for a clear human answer
        return f"""
        The user asked: "{user_message}"

        The structured spec was:
        {spec}

        The query returned these rows:
        {results}

        Write a natural language response summarizing the answer clearly.
        """
    
    def _extract_json(self, raw_text: str) -> dict:
        # Strip markdown or text around JSON
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM response")
        return json.loads(match.group(0))

    def get_player_id(self, name: str) -> int:
            """Fetch the ID from the Player table using the player's name."""
            player = self.db_session.query(Player).filter(Player.name == name).first()
            if not player or not player.id:
                raise ValueError(f"Player not found or missing MLBAM ID: {name}")
            return player.id
    
    def spec_to_query(self, spec: dict):
        select_fields = spec.get("select")
        filters = spec.get("filters", {})
        aggregate = spec.get("aggregate")
        order_by = spec.get("order_by")
        limit = spec.get("limit")

        if isinstance(select_fields, str):
            select_fields = [select_fields]

        query = self.db_session.query()
        joins = set()
        columns = []

        # Extract team filter early (for computed aliases)
        team_name = filters.get("game_filters", {}).get("team")

        # --- Computed Aliases ---
        alias_map = {
            "runs_scored": case(
                (Game.home_team == team_name, Game.home_score),
                else_=Game.away_score
            ).label("runs_scored"),
            "opponent_runs": case(
                (Game.home_team == team_name, Game.away_score),
                else_=Game.home_score
            ).label("opponent_runs"),
            "date": Game.date
        }

        def resolve_value(val):
            """Convert filter value into a column/expression if it's an alias."""
            if isinstance(val, str) and val in alias_map:
                return alias_map[val]
            return val

        # --- Handle select fields ---
        for field in select_fields:
            if field in alias_map:
                col = alias_map[field]
            elif hasattr(Game, field):
                col = getattr(Game, field)
            elif hasattr(Weather, field):
                joins.add("weather")
                col = getattr(Weather, field)
            elif hasattr(PlayerGameStats, field):
                joins.add("playergamestats")
                col = getattr(PlayerGameStats, field)
            elif hasattr(Player, field):
                joins.add("player")
                col = getattr(Player, field)
            else:
                raise ValueError(f"Unknown select field: {field}")


            columns.append(col)

        query = query.with_entities(*columns).select_from(Game)

        # --- Handle joins ---
        if "weather" in joins:
            query = query.join(Weather, Weather.game_id == Game.id)

        if "playergamestats" in joins or "player_filters" in filters or "player" in joins:
            query = query.join(PlayerGameStats, PlayerGameStats.game_id == Game.id)
            query = query.join(Player, Player.id == PlayerGameStats.player_id)

        # --- Apply filters ---
        conditions = []

        def add_condition(col, op, val):
            if op == "eq":
                return col == val
            elif op == "gt":
                return col > val
            elif op == "lt":
                return col < val
            elif op == "gte":
                return col >= val
            elif op == "lte":
                return col <= val
            else:
                raise ValueError(f"Unsupported operator: {op}")

        # Game filters
        for key, val in filters.get("game_filters", {}).items():
            if key == "team":
                # Match either home or away
                conditions.append(or_(Game.home_team == val, Game.away_team == val))

            elif key == "opponent_team":
                # Match games where the opponent is the given team
                conditions.append(or_(
                    and_(Game.home_team == val, Game.away_team != val),
                    and_(Game.away_team == val, Game.home_team != val),
                ))

            elif key in alias_map:
                col = alias_map[key]
                conditions.append(col == resolve_value(val))

            elif hasattr(Game, key):
                conditions.append(getattr(Game, key) == resolve_value(val))

            elif key.endswith(("_gt", "_lt", "_gte", "_lte")):
                base_key, suffix = key.rsplit("_", 1)
                if base_key in alias_map:
                    col = alias_map[base_key]
                elif hasattr(Game, base_key):
                    col = getattr(Game, base_key)
                else:
                    raise ValueError(f"Unknown game filter: {base_key}")
                conditions.append(add_condition(col, suffix, resolve_value(val)))

            else:
                raise ValueError(f"Unknown game filter: {key}")

        # Player filters
        for key, val in filters.get("player_filters", {}).items():
            if hasattr(Player, key):
                conditions.append(getattr(Player, key) == val)
            else:
                raise ValueError(f"Unknown player filter: {key}")

        if conditions:
            query = query.filter(and_(*conditions))

        # --- Ordering ---
        if order_by == "date_desc":
            query = query.order_by(desc(Game.date))
        elif order_by == "date_asc":
            query = query.order_by(asc(Game.date))

        # --- Limit ---
        if limit:
            query = query.limit(limit)

        # --- aggregation ---
        if aggregate:
            agg_map = {
                "sum": func.sum,
                "avg": func.avg,
                "max": func.max,
                "min": func.min,
            }
            if aggregate not in agg_map:
                raise ValueError(f"Unsupported aggregate: {aggregate}")

            # Wrap the limited & ordered query as a subquery
            subq = query.subquery()
            agg_columns = [agg_map[aggregate](getattr(subq.c, c.key)).label(c.key) for c in subq.c]
            query = self.db_session.query(*agg_columns)

        # ---- Compile SQL for debugging ----
        compiled = query.statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True}
        )
        print(compiled)

        return query
    
    def calculate_results(self, spec: dict, results: list) -> list:
        calculate = spec.get("calculate")
        if not calculate or not results:
            return results
        row = results[0]

        if calculate == "AVG":
            # Results are in alphabetical order'
            total_at_bats = row[0]
            total_hits = row[1]
            batting_average = total_hits / total_at_bats if total_at_bats > 0 else 0
            return [{"batting_average": round(batting_average, 3)}]

        elif calculate == "ERA":
            # Assuming results contain summed 'earned_runs' and 'outs_pitched'
            total_earned_runs = row[0]
            total_outs_pitched = row[1]
            if total_outs_pitched > 0:
                innings_pitched = total_outs_pitched / 3
                era = (total_earned_runs * 9) / innings_pitched
            else:
                era = 0
            return [{"ERA": round(era, 2)}]
        
        elif calculate == "WHIP":
            # Alphabetical: hits_allowed, outs_pitched, walks
            total_hits_allowed = row[0]
            total_outs_pitched = row[1]
            total_walks = row[2]
            innings_pitched = total_outs_pitched / 3 if total_outs_pitched > 0 else 0
            whip = (total_hits_allowed + total_walks) / innings_pitched if innings_pitched > 0 else 0
            return [{"WHIP": round(whip, 3)}]

        elif calculate == "OBP":
            # Alphabetical: at_bats, hits, walks_batting
            total_at_bats = row[0]
            total_hits = row[1]
            total_walks_batting = row[2]
            denominator = total_at_bats + total_walks_batting
            obp = (total_hits + total_walks_batting) / denominator if denominator > 0 else 0
            return [{"OBP": round(obp, 3)}]

        # Slugging Percentage (SLG): (1B + 2*2B + 3*3B + 4*HR) / at_bats
        elif calculate == "SLG":
            # Alphabetical: at_bats, doubles, hits, home_runs, triples
            total_at_bats = row[0]
            total_doubles = row[1]
            total_hits = row[2]
            total_home_runs = row[3]
            total_triples = row[4]
            singles = total_hits - total_doubles - total_triples - total_home_runs
            total_bases = singles + 2 * total_doubles + 3 * total_triples + 4 * total_home_runs
            slg = total_bases / total_at_bats if total_at_bats > 0 else 0
            return [{"SLG": round(slg, 3)}]

        elif calculate == "OPS":
            # Alphabetical: at_bats, doubles, hits, home_runs, triples, walks_batting
            total_at_bats = row[0]
            total_doubles = row[1]
            total_hits = row[2]
            total_home_runs = row[3]
            total_triples = row[4]
            total_walks_batting = row[5]
            singles = total_hits - total_doubles - total_triples - total_home_runs

            # OBP
            obp_denominator = total_at_bats + total_walks_batting
            obp = (total_hits + total_walks_batting) / obp_denominator if obp_denominator > 0 else 0

            # SLG
            total_bases = singles + 2 * total_doubles + 3 * total_triples + 4 * total_home_runs
            slg = total_bases / total_at_bats if total_at_bats > 0 else 0

            ops = obp + slg
            return [{"OPS": round(ops, 3)}]

        else:
            raise ValueError(f"Unsupported calculation: {calculate}")

    def handle_request(self, state: GraphState):
        """
        End-to-end: natural language request -> JSON spec -> SQL -> DB results -> natural language answer.
        """
        user_message = state["input"]
        # --- Step 1. Ask LLM to generate JSON spec ---
        prompt = self._build_spec_prompt(user_message)
        spec_resp = self.openai_client.chat(
            [
                {"role": "system", "content": "You are a query translator that outputs ONLY valid JSON."},
                {"role": "user","content": prompt}
            ],
            model="gpt-5"
        )
        
        spec = _extract_json(spec_resp)
        if "select" in spec and isinstance(spec["select"], list):
            spec["select"].sort()
        print(spec)

        filters = spec.get("filters", {})
        at_bat_filters = filters.get("at_bat_filters")
        if at_bat_filters:
            select_fields = spec.get("select")
            aggregate = spec.get("aggregate")
            limit = spec.get("limit")
            player_name = filters.get("player_filters", {}).get("name")
            player_id = self.get_player_id(player_name)
            position = at_bat_filters.get("player_type")
            opponent_name = at_bat_filters.get("opposing_player", None)
            opponent_id = self.get_player_id(opponent_name)
            season = filters.get("game_filters", {}).get("season_year", None)

            print(
                f"[DEBUG] get_matchup_stats call → "
                f"player_name={player_name}, player_id={player_id}, "
                f"position={position}, opponent_name={opponent_name}, opponent_id={opponent_id}, "
                f"season={season}, limit={limit}, select_fields={select_fields}, aggregate={aggregate}"
            )

            # Redirect out to external service
            results =  self.atbat_service.get_matchup_stats(
                player_id=player_id,
                position=position,
                opponent_id=opponent_id,
                limit=limit,
                season=season,
                select_fields=select_fields,
                aggregate=aggregate
            )
        else: 
            # # --- Step 2. Run the SQL query ---
            query = self.spec_to_query(spec)
            results = query.all()
        print(f"raw results: {results}")
        calculated_results = self.calculate_results(spec, results)
        print(f"calculated Results: {calculated_results}")
        # --- Step 3. Ask LLM to format results for user ---
        answer_prompt = self._build_answer_prompt(user_message, spec, calculated_results)
        result = self.openai_client.chat(
            [
                {
                    "role": "system",
                    "content": "You are a helpful sports data assistant. Convert query results into clear, natural language answers."
                },
                {
                    "role": "user",
                    "content": answer_prompt
                }
            ]
        )
        return {**state, "output": result}
    

if __name__ == "__main__":
    atbat_service = AtBatService()
    agent = StatsAgent(db_session=db, openai_client=openai_client, atbat_service=atbat_service)
    # res = agent.handle_request("How many home runs does Shohei Ohtani have this season?")
    state = GraphState(input="What is shohei's batting average against tarik skubal this season?", intent="STAT")
    # state = GraphState(input="How did Kevin Gausman do in his last 5 at bats against Aaron Judge?", intent="STAT")
    # state = GraphState(input="What is Shohei's batting average this year?", intent="STAT")
    res = agent.handle_request(state)
    print(res)  