from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

from website_chatbot.chunking import RecursiveTextChunker
from website_chatbot.config import Settings
from website_chatbot.crawler import Crawl4AICrawler
from website_chatbot.embedding import SentenceTransformerEmbedder
from website_chatbot.pipeline import IngestionPipeline
from website_chatbot.sites import SiteIngestion, build_site_ingestions
from website_chatbot.storage import QdrantVectorStore


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class IngestionOptions:
    urls: list[str] | None = None
    websites_file: str | None = None
    urls_file: str | None = None  # deprecated: use websites_file
    site: str | None = None
    output: str | None = None
    deep_crawl: bool | None = None
    css_selector: str | None = None
    url_contains: str | None = None
    model: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    qdrant: bool = False
    qdrant_recreate: bool | None = None
    verbose: bool | None = None

    @classmethod
    def from_env_defaults(cls, websites_file: str | Path) -> IngestionOptions:
        """Build options for ``python main.py`` using .env + websites.json."""
        return cls(
            websites_file=str(websites_file),
            output=os.getenv("OUTPUT_JSON") or None,
            qdrant=_env_bool("SAVE_TO_QDRANT", True),
            qdrant_recreate=_env_bool("QDRANT_RECREATE_COLLECTION", False),
        )


def _embedded_to_dict(item) -> dict:
    return {
        "chunk_id": item.chunk.chunk_id,
        "text": item.chunk.text,
        "source_url": item.chunk.source_url,
        "title": item.chunk.title,
        "chunk_index": item.chunk.chunk_index,
        "website": item.chunk.metadata.get("website", ""),
        "metadata": item.chunk.metadata,
        "model_name": item.model_name,
        "vector": item.vector,
    }


def _output_path_for_site(base: Path, job: SiteIngestion, multi_site: bool) -> Path:
    if not multi_site:
        return base
    return base.parent / f"{base.stem}_{job.website_name}{base.suffix}"


async def _ingest_site(
    job: SiteIngestion,
    settings: Settings,
    *,
    use_qdrant: bool,
    embedder: SentenceTransformerEmbedder,
) -> list:
    qdrant_cfg = settings.qdrant.for_website(job.website_name)
    vector_store = QdrantVectorStore(qdrant_cfg) if use_qdrant else None

    pipeline = IngestionPipeline(
        crawler=Crawl4AICrawler(settings.crawler),
        chunker=RecursiveTextChunker(settings.chunker),
        embedder=embedder,
        vector_store=vector_store,
    )

    collection = qdrant_cfg.collection_name
    print(
        f"\n--- {job.website_name} ({len(job.urls)} URL(s)) "
        f"→ collection '{collection}' ---"
    )

    if vector_store:
        embedded = await pipeline.run_and_store(
            job.urls, website_name=job.website_name
        )
        print(f"Stored {len(embedded)} chunk(s) in '{collection}'")
    else:
        embedded = await pipeline.run(job.urls, website_name=job.website_name)
        print(f"Embedded {len(embedded)} chunk(s)")

    return embedded


async def run_ingestion(options: IngestionOptions) -> int:
    jobs = build_site_ingestions(
        cli_urls=options.urls,
        websites_file=options.websites_file,
        urls_file=options.urls_file,
        site_override=options.site,
    )
    if not jobs:
        print("Error: no websites/URLs found. Check your websites.json file.")
        return 1

    settings = Settings.from_env()

    if options.deep_crawl is True:
        settings.crawler.deep_crawl = True
    if options.css_selector:
        settings.crawler.css_selector = options.css_selector
    if options.url_contains:
        settings.crawler.url_contains = options.url_contains
    if options.verbose is True:
        settings.crawler.verbose = True
    if options.model:
        settings.embedder.model_name = options.model
    if options.chunk_size:
        settings.chunker.chunk_size = options.chunk_size
    if options.chunk_overlap:
        settings.chunker.chunk_overlap = options.chunk_overlap
    if options.qdrant_recreate:
        settings.qdrant = replace(settings.qdrant, recreate_collection=True)

    embedder = SentenceTransformerEmbedder(settings.embedder)
    multi_site = len(jobs) > 1
    all_embedded = []

    print(
        f"Ingesting {len(jobs)} website(s): "
        + ", ".join(j.website_name for j in jobs)
    )

    for job in jobs:
        embedded = await _ingest_site(
            job, settings, use_qdrant=options.qdrant, embedder=embedder
        )
        all_embedded.extend(embedded)

        if options.output:
            out_path = _output_path_for_site(Path(options.output), job, multi_site)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = [_embedded_to_dict(item) for item in embedded]
            out_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Wrote {out_path}")

    print(
        f"\nDone: {len(all_embedded)} total chunk(s) across {len(jobs)} website(s) "
        f"({settings.embedder.model_name}, dim={embedder.vector_dimensions})"
    )
    if options.qdrant:
        print(f"Qdrant: {settings.qdrant.url}")

    return 0
