import os
import requests
from app.interfaces.i_news_client import INewsClient

GO_NEWS_URL = os.getenv("GO_NEWS_URL", "http://localhost:8092")


class GoNewsClient(INewsClient):

    def __init__(self, base_url: str = GO_NEWS_URL):
        self.base_url = base_url.rstrip("/")

    def scrape(self) -> dict:
        resp = requests.post(
            f"{self.base_url}/scrape",
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
