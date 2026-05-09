from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel

from app.graph.workflow import build_graph

router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@lru_cache(maxsize=1)
def _get_graph():
    return build_graph()


@router.post("", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    graph = _get_graph()
    result = graph.invoke({"input": body.message})
    return ChatResponse(response=result.get("output", "No response"))
