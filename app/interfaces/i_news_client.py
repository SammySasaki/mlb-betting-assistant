from abc import ABC, abstractmethod


class INewsClient(ABC):

    @abstractmethod
    def scrape(self) -> dict:
        """Call the Go news service and return the raw response dict.

        Expected keys: articles (list), sources_scraped (int),
        total_articles (int), scrape_duration_ms (int).
        """
        pass
