from website_chatbot.crawler.base import BaseCrawler

__all__ = ["BaseCrawler", "Crawl4AICrawler"]


def __getattr__(name: str):
    if name == "Crawl4AICrawler":
        from website_chatbot.crawler.crawl4ai import Crawl4AICrawler

        return Crawl4AICrawler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
