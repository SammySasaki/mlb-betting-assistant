import json
from datetime import date
from app.graph.state import GraphState
from app.interfaces.illm_client import ILLMClient
from app.interfaces.ilineup_repository import ILineupRepository
from app.interfaces.iplayer_repository import IPlayerRepository
from app.interfaces.igame_repository import IGameRepository
from app.services.lineup_service import LineupService

class LineupAgent:
    def __init__(self, player_repository: IPlayerRepository, 
                 lineup_repository: ILineupRepository, 
                 llm_client: ILLMClient,
                 lineup_service: LineupService,
                 game_repository: IGameRepository):
        self.player_repository = player_repository
        self.llm_client = llm_client
        self.lineup_repository = lineup_repository
        self.lineup_service = lineup_service
        self.game_repository = game_repository

    def _build_prompt(self, user_message: str) -> str:
        """
        Build a prompt for the LLM using the COSTAR method, instructing it to return a JSON spec
        describing how to filter and extract information from the response of get_lineups_by_date,
        based on the user's question about today's lineups.
        """

        costar_context = """
        You are an expert baseball data assistant.
        Your task is to interpret user questions about today's MLB lineups and return a JSON specification
        that describes how to filter and extract information from the response of get_lineups_by_date(game_date).

        Do NOT specify which repository method to call. Instead, describe what to do with the response from get_lineups_by_date.
        The response will be a list of Lineups objects for all games on a given date.

        Batting orders are in the format 100, 200, ..., 900 for positions 1-9.

        COSTAR Method:
        C - Context: The user is asking about today's MLB lineups.
        O - Objective: Return a JSON spec describing how to filter and extract the relevant information.
        S - Structure: Use the following JSON format:
            {
                "filters": { ... },
                "extract": "<description of what to extract>",
                "reasoning": "<short reasoning>"
            }
        T - Task: Map the user's question to the correct filters and extraction instructions.
        A - Actions: Only use the response from get_lineups_by_date.
        R - Reasoning: Briefly explain your reasoning.

        Today's date is available as {today_date}.

        Examples:
        User: "Who is batting 1st for the Yankees today?"
        JSON:
        {
            "filters": { "team": "New York Yankees", "batting_order": 100 },
            "extract": "player name",
            "reasoning": "Filter for New York Yankees and batting_order 1 to find the leadoff hitter."
        }

        User: "Give me the whole lineup for the Dodgers today."
        JSON:
        {
            "filters": { "team": "Los Angeles Dodgers" },
            "extract": "all lineup entries for the Dodgers, ordered by batting_order",
            "reasoning": "Filter for Los Angeles Dodgers to get the full lineup."
        }

        User: "Who is the SS for the Boston Red Sox today?"
        JSON:
        {
            "filters": { "team": "Boston Red Sox", "defensive_position": "SS" },
            "extract": "player name",
            "reasoning": "Filter for Boston Red Sox and SS position to find the shortstop."
        }

        User: "Who is starting for the Cubs today?"
        JSON:
        {
            "filters": { "team": "Chicago Cubs", "defensive_position": "SP" },
            "extract": "player name",
            "reasoning": "Filter for Chicago Cubs and SP postition to find the starting pitcher."
        }

        Now, given the following user message, return only the JSON spec as described above.
        User message: \"{user_message}\"
        """

        from datetime import date
        today_date = str(date.today())

        prompt = costar_context.replace("{today_date}", today_date).replace("{user_message}", user_message)
        return prompt

    def _read_json(self, llm_response):
        """
        Parse the LLM JSON response and call get_lineups_by_date_by_team with appropriate filters.
        Further filter by defensive_position or batting_order if specified.
        """
        spec = json.loads(llm_response)
        print(spec)
        filters = spec.get("filters", {})
        defensive_position = filters.get("defensive_position", "")
        team = filters.get("team")
        game = self.game_repository.get_game_by_date_by_team(date.today(), team) if team else None
        if game:
            self.lineup_service.save_lineups_for_game(game.id)
        game_date = date.today()  # or pass as argument if needed

        # Special handling for starting pitcher (SP)
        if defensive_position == "SP" and team:
            game = self.game_repository.get_game_by_date_by_team(game_date, team)
            if not game:
                return [{"error": f"No game found for {team} on {game_date}"}]
            # Determine if team is home or away
            if game.home_team == team:
                pitcher_id = game.home_probable_pitcher_id
            elif game.away_team == team:
                pitcher_id = game.away_probable_pitcher_id
            else:
                pitcher_id = None
            player_name = self.player_repository.get_player_name(pitcher_id) if pitcher_id else None
            return [{
                "player_id": pitcher_id,
                "player_name": player_name,
                "team": team,
                "role": "SP"
            }]

        # Get initial lineup entries for the team and date
        lineups = self.lineup_repository.get_lineups_by_date_by_team(game_date, team) if team else []

        # Further filter by defensive_position and/or batting_order
        if "defensive_position" in filters:
            lineups = [l for l in lineups if l.defensive_position == filters["defensive_position"]]
        if "batting_order" in filters:
            lineups = [l for l in lineups if l.batting_order == filters["batting_order"]]

        # Extract requested info
        extract = spec.get("extract", "")
        result = []
        for l in lineups:
            player_name = self.player_repository.get_player_name(l.player_id)
            result.append({
                "player_name": player_name,
                "batting_order": l.batting_order,
                "defensive_position": l.defensive_position
            })
            # Add more extraction logic as needed

        return result


    def handle_request(self, state: GraphState):
        """
        End-to-end: natural language request -> JSON spec -> SQL -> DB results -> natural language answer.
        """
        user_message = state["input"]
        # --- Step 1. Ask LLM to generate JSON spec ---
        prompt = self._build_prompt(user_message)
        llm_response = self.llm_client.chat(
            [
                {"role": "system", "content": "You are a query translator that outputs ONLY valid JSON."},
                {"role": "user","content": prompt}
            ],
            model="gpt-5"
        )
        result = self._read_json(llm_response)
        return {**state, "output": result}
    

    # Who is batting 1st for the Yankees today?