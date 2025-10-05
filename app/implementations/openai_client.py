from openai import OpenAI
from typing import List, Dict
from app.interfaces.illm_client import ILLMClient


class OpenAILLMClient(ILLMClient):
    """Concrete implementation of ILLMClient using OpenAI."""

    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message.content.strip()