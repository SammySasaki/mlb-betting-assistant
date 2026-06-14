from openai import OpenAI
from typing import Dict, Any
import os
from app.graph.state import GraphState
from app.interfaces.illm_client import ILLMClient

CATEGORIES = ["STAT", "LIVE_STAT", "RECOMMENDATION", "LINEUP", "OBSERVATION", "NEWS", "OTHER"]
SYSTEM_PROMPT = "You are a sports betting analyst assistant."

class ClassifierAgent:

    def __init__(self, api_client: ILLMClient, model="gpt-5"):
        self.client = api_client
        self.model = model
        self.system_prompt = "You are a sports betting analyst assistant."

    def _build_prompt(self, user_message: str) -> str:
        return f"""
        Classify the following user message into one of:
        {", ".join(CATEGORIES)}

        Use LIVE_STAT for questions about a player's season-level stats or league leaderboards
        (e.g. ERA leaders, home run totals, batting average for a specific player this season).

        Use STAT for questions that require game-level detail, trends across multiple games,
        weather correlations, or head-to-head records that span individual game rows.

        Examples:
        - "Who leads the league in ERA?" → LIVE_STAT
        - "Top 10 home run leaders this season" → LIVE_STAT
        - "What is Shohei Ohtani's batting average this year?" → LIVE_STAT
        - "How many strikeouts does Freddy Peralta have this season?" → LIVE_STAT
        - "How many runs per game did the Giants score in their last 5 road games?" → STAT
        - "What is Shohei Ohtani's ERA in games with wind over 15mph?" → STAT
        - "How did the Giants score in each of their last 3 home games?" → STAT
        - "Should I bet the over for tonight's Yankees game?" → RECOMMENDATION
        - "The Dodgers have been hitting really well lately." → OBSERVATION
        - "Who is batting 1st for the Yankees today?" → LINEUP
        - "Who is playing shortstop for the Dodgers today?" → LINEUP
        - "Who is starting for the Giants today?" → LINEUP
        - "Any injury news on the Dodgers?" → NEWS
        - "What happened with Ohtani?" → NEWS
        - "Any lineup changes today?" → NEWS
        - "Latest transactions this week?" → NEWS

        Message: "{user_message}"
        Answer with just the category name.
        """

    def classify_message(self, state: GraphState): 
        """Ask LLM to classify the intent of the user's message""" 
        user_message = state["input"] 
        intent = "" 
        categories = CATEGORIES
        raw_output = self.client.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self._build_prompt(user_message)},
                ],
                model=self.model,
            )
        raw_output = raw_output.upper().strip()
        # clean: handle cases like "Category: STAT" or "STAT\n" 
        for cat in categories: 
            if cat in raw_output: 
                intent = cat 
            if intent == "": 
                intent = "OTHER"

        print(intent) 
        
        return {**state, "intent": intent}

if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    agent = ClassifierAgent(client)
    state = GraphState(input="What should I bet today for the Giants game?")
    