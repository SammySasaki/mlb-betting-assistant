from abc import ABC, abstractmethod
from typing import List, Dict


class ILLMClient(ABC):
    """Abstract interface for any LLM provider."""

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], model: str) -> str:
        """Send chat messages and return response text."""
        pass