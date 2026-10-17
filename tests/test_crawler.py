from __future__ import annotations

from website_chatbot.config import CrawlerConfig
from website_chatbot.crawler.crawl4ai import Crawl4AICrawler


def test_css_selector_targets_content_without_dropping_page_metadata():
    run_config = Crawl4AICrawler(CrawlerConfig(css_selector="main"))._build_run_config()

    assert run_config.target_elements == ["main"]
    assert run_config.css_selector is None
