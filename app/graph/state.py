from typing import TypedDict, Optional, List

class GraphState(TypedDict, total=False):
    input: str
    intent: Optional[str]
    output: Optional[str]
    predictions_data: Optional[List[dict]]
    context_data: Optional[dict]
