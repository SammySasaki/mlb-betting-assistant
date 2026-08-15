import json
from sqlalchemy.orm import Session

from app.graph.state import GraphState
from app.interfaces.illm_client import ILLMClient
from app.implementations.sqlalchemy_news_repository import SqlAlchemyNewsRepository
from app.utils.utils import llm_call_with_retry


class NewsAgent:

    def __init__(self, openai_client: ILLMClient, db_session: Session):
        self.openai_client = openai_client
        self.repo = SqlAlchemyNewsRepository(db_session)

    def _build_spec_prompt(self, user_message: str) -> str:
        return f"""
        You are a query translator for an MLB news database.

        Extract a filter spec from the user's question. Return valid JSON only.

        ### Response Format
        {{
            "teams": ["<full team name>", ...],   // empty list if no specific team mentioned
            "hours": <int>,                        // how far back to look (default 24)
            "category": "<INJURY|LINEUP|TRANSACTION|GENERAL|null>"  // null = any
        }}

        ### Examples
        "Any injury news on the Dodgers?" →
        {{"teams": ["Los Angeles Dodgers"], "hours": 24, "category": "INJURY"}}

        "What lineup changes happened today?" →
        {{"teams": [], "hours": 24, "category": "LINEUP"}}

        "Latest news on the Yankees or Red Sox" →
        {{"teams": ["New York Yankees", "Boston Red Sox"], "hours": 48, "category": null}}

        "Any transactions this week?" →
        {{"teams": [], "hours": 168, "category": "TRANSACTION"}}

        ### Task
        "{user_message}"
        """

    def handle_request(self, state: GraphState) -> GraphState:
        user_message = state["input"]

        spec = llm_call_with_retry(
            self.openai_client,
            messages=[{"role": "system", "content": self._build_spec_prompt(user_message)}],
        )

        teams = spec.get("teams") or []
        hours = int(spec.get("hours") or 24)
        category = spec.get("category") or None

        if teams:
            items = self.repo.get_recent_by_teams(teams, hours=hours)
        else:
            items = self.repo.get_recent(limit=15, category=category)

        if not items:
            output = "No recent news found matching your query."
        else:
            lines = []
            for item in items:
                ts = item.published_at.strftime("%b %d %H:%M") if item.published_at else "recent"
                lines.append(f"[{item.source.upper()} – {ts}] {item.headline}")
            output = "\n".join(lines)

        return {**state, "output": output}
