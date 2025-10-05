from typing import TypedDict, Optional

class GraphState(TypedDict, total=False):
    input: str             # raw user message
    intent: Optional[str]  # classifier decision (STAT, RECOMMENDATION, etc)
    output: Optional[str]  # final agent response
