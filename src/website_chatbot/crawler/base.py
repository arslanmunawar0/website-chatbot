from __future__ import annotations

from abc import ABC, abstractmethod

from website_chatbot.models import Document


class BaseCrawler(ABC):
    """Fetch web pages and normalize them into documents."""

    @abstractmethod
    async def crawl(self, urls: list[str]) -> list[Document]:
        """Crawl one or more seed URLs and return extracted documents."""
