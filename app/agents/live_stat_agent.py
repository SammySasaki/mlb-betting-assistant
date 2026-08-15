"""
LiveStatAgent — routes player season stats and league leader queries
directly to the MLB Stats API rather than aggregating from the local DB.
"""
from app.graph.state import GraphState
from app.utils.utils import llm_call_with_retry
from app.interfaces.illm_client import ILLMClient
from app.implementations.mlb_api_client import StatsApiClient
from app.interfaces.iplayer_repository import IPlayerRepository

# Maps natural-language stat names to MLB Stats API leaderCategories values.
# Used in the prompt so the LLM picks the right key.
STAT_CATEGORY_MAP = """
ERA / earned run average    → earnedRunAverage       (pitching)
WHIP                        → walksAndHitsPerInningPitched (pitching)
strikeouts / Ks             → strikeouts             (pitching)
wins                        → wins                   (pitching)
saves                       → saves                  (pitching)
innings pitched             → inningsPitched         (pitching)
batting average / AVG       → battingAverage         (hitting)
home runs / HR              → homeRuns               (hitting)
RBIs / runs batted in       → runsBattedIn           (hitting)
hits                        → hits                   (hitting)
runs                        → runs                   (hitting)
stolen bases / SB           → stolenBases            (hitting)
OBP / on-base percentage    → onBasePercentage       (hitting)
OPS                         → onBasePlusSlugging     (hitting)
slugging / SLG              → sluggingPercentage     (hitting)
walks / BB                  → baseOnBalls            (hitting)
"""


class LiveStatAgent:
    def __init__(self, mlb_client: StatsApiClient, llm_client: ILLMClient, player_repo: IPlayerRepository, season: int = 2026):
        self.mlb_client = mlb_client
        self.llm_client = llm_client
        self.player_repo = player_repo
        self.season = season

    def _extract_spec(self, user_message: str) -> dict:
        prompt = f"""
Convert this MLB stats question into a JSON spec.

Stat category mappings:
{STAT_CATEGORY_MAP}

Output JSON with these fields:
{{
  "query_type": "leader" | "player",
  "stat_category": "<MLB API category from the map above>",
  "stat_group": "pitching" | "hitting",
  "season": <year, default {self.season}>,
  "player_name": "<full name if a specific player, else null>",
  "limit": <int, default 10 for leaders>
}}

Examples:
Q: "Who leads the league in ERA?"
{{"query_type":"leader","stat_category":"earnedRunAverage","stat_group":"pitching","season":{self.season},"player_name":null,"limit":10}}

Q: "Top 5 home run leaders this season"
{{"query_type":"leader","stat_category":"homeRuns","stat_group":"hitting","season":{self.season},"player_name":null,"limit":5}}

Q: "What is Shohei Ohtani's batting average this year?"
{{"query_type":"player","stat_category":"battingAverage","stat_group":"hitting","season":{self.season},"player_name":"Shohei Ohtani","limit":null}}

Q: "How many strikeouts does Freddy Peralta have?"
{{"query_type":"player","stat_category":"strikeouts","stat_group":"pitching","season":{self.season},"player_name":"Freddy Peralta","limit":null}}

Now convert: "{user_message}"
Output only valid JSON.
"""
        return llm_call_with_retry(
            self.llm_client,
            messages=[
                {"role": "system", "content": "Output only valid JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o-mini",
        )

    def _format_answer(self, user_message: str, spec: dict, data) -> str:
        prompt = f"""
The user asked: "{user_message}"
The query spec was: {spec}
The data returned: {data}

Write a concise, natural language answer. For leaderboards include rank, name, and stat value.
"""
        return self.llm_client.chat(
            [
                {"role": "system", "content": "You are a helpful MLB stats assistant."},
                {"role": "user", "content": prompt},
            ]
        )

    def handle_request(self, state: GraphState) -> GraphState:
        user_message = state["input"]
        spec = self._extract_spec(user_message)

        season = spec.get("season") or self.season
        stat_category = spec["stat_category"]
        stat_group = spec.get("stat_group", "hitting")
        limit = spec.get("limit") or 10

        if spec["query_type"] == "leader":
            data = self.mlb_client.get_league_leaders(
                stat_category=stat_category,
                season=season,
                stat_group=stat_group,
                limit=limit,
            )
        else:
            player_name = spec.get("player_name")
            if not player_name:
                return {**state, "output": "I couldn't determine which player you're asking about."}

            player = self.player_repo.get_by_name(player_name)
            if not player:
                return {**state, "output": f"I couldn't find '{player_name}' in the database."}

            data = self.mlb_client.get_player_season_stats(
                player_id=player.id,
                season=season,
                stat_group=stat_group,
            )
            if not data:
                return {**state, "output": f"No {season} season stats found for {player_name}."}

        answer = self._format_answer(user_message, spec, data)
        return {**state, "output": answer}
