from openai import OpenAI
from typing import Dict, Any
import os
from app.graph.state import GraphState
from app.interfaces.illm_client import ILLMClient

CATEGORIES = ["STAT", "RECOMMENDATION", "LINEUP", "OBSERVATION", "OTHER"]
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

        Examples:
        - "How many runs per game did the Giants score in their last 5 road games?" → STAT
        - "What is Shohei Ohtani's ERA over the last 5 games against the Red Sox" → STAT
        - "Should I bet the over for tonight's Yankees game?" → RECOMMENDATION
        - "What is the latest news on the Mets?" → NEWS
        - "The Dodgers have been hitting really well lately." → OBSERVATION
        - "Who is batting 1st for the Yankees today?" → LINEUP
        - "Who is playing shortstop for the Dodgers today?" → LINEUP
        - "Who is starting for the Giants today?" → LINEUP

        Message: "{user_message}"
        Answer with just the category name.
        """

    def classify_message(self, state: GraphState): 
        """Ask LLM to classify the intent of the user's message""" 
        user_message = state["input"] 
        intent = "" 
        categories = ["STAT", "RECOMMENDATION", "LINEUP", "OBSERVATION", "OTHER"]
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
    