from __future__ import annotations

from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy

from website_chatbot.config import CrawlerConfig
from website_chatbot.crawler.base import BaseCrawler
from website_chatbot.models import Document


class Crawl4AICrawler(BaseCrawler):
    """Crawl pages with Crawl4AI — single URLs or optional BFS deep crawl."""

    def __init__(self, config: CrawlerConfig | None = None) -> None:
        self.config = config or CrawlerConfig()

    async def crawl(self, urls: list[str]) -> list[Document]:
        if not urls:
            return []

        browser_config = BrowserConfig(headless=self.config.headless)
        run_config = self._build_run_config()

        documents: list[Document] = []
        seen_urls: set[str] = set()

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for seed_url in urls:
                results = await crawler.arun(url=seed_url, config=run_config)
                for result in self._iter_results(results):
                    if result.url in seen_urls:
                        continue
                    seen_urls.add(result.url)

                    doc = self._result_to_document(result)
                    if doc is not None:
                        documents.append(doc)
                        if self.config.verbose:
                            print("Crawled:", result.url)

        return documents

    def _build_run_config(self) -> CrawlerRunConfig:
        kwargs: dict[str, Any] = {"verbose": self.config.verbose}
        if self.config.css_selector:
            kwargs["css_selector"] = self.config.css_selector
        if self.config.deep_crawl:
            kwargs["deep_crawl_strategy"] = BFSDeepCrawlStrategy(
                max_depth=self.config.max_depth,
                include_external=self.config.include_external,
            )
        return CrawlerRunConfig(**kwargs)

    @staticmethod
    def _iter_results(results: Any):
        if results is None:
            return
        if isinstance(results, list):
            yield from results
        else:
            yield results

    def _result_to_document(self, result: Any) -> Document | None:
        if not getattr(result, "success", False):
            if self.config.verbose:
                print("FAILED:", getattr(result, "url", "?"))
            return None

        url = result.url
        if self.config.url_contains and self.config.url_contains not in url:
            return None

        content = (result.markdown or "").strip()
        if len(content) < self.config.min_content_length:
            return None

        metadata = dict(getattr(result, "metadata", None) or {})
        title = metadata.get("title", "") or ""

        return Document(
            url=url,
            title=title,
            content=content,
            metadata={"crawler": "crawl4ai", **metadata},
        )
