from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None else default


@dataclass(slots=True)
class CrawlerConfig:
    headless: bool = True
    css_selector: str | None = None
    min_content_length: int = 100
    url_contains: str | None = None
    deep_crawl: bool = False
    max_depth: int = 2
    include_external: bool = False
    verbose: bool = False


@dataclass(slots=True)
class ChunkerConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass(slots=True)
class EmbedderConfig:
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    normalize_embeddings: bool = True


@dataclass(slots=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    api_key: str | None = None
    collection_name: str = "website_chunks"
    collection_prefix: str = ""
    vector_size: int | None = None
    recreate_collection: bool = False

    def for_website(self, website_name: str) -> QdrantConfig:
        """Qdrant settings with collection name derived from the website."""
        from website_chatbot.sites import collection_name_for_website

        return QdrantConfig(
            url=self.url,
            api_key=self.api_key,
            collection_name=collection_name_for_website(
                website_name, prefix=self.collection_prefix
            ),
            collection_prefix=self.collection_prefix,
            vector_size=self.vector_size,
            recreate_collection=self.recreate_collection,
        )


@dataclass(slots=True)
class ChatConfig:
    llm_model: str = "google_genai:gemini-3.5-flash-lite"
    temperature: float = 0.0
    top_k: int = 5
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173"]
    )


@dataclass(slots=True)
class Settings:
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            crawler=CrawlerConfig(
                headless=_env_bool("CRAWLER_HEADLESS", True),
                css_selector=os.getenv("CRAWLER_CSS_SELECTOR"),
                min_content_length=_env_int("CRAWLER_MIN_CONTENT_LENGTH", 100),
                url_contains=os.getenv("CRAWLER_URL_CONTAINS"),
                deep_crawl=_env_bool("CRAWLER_DEEP_CRAWL", False),
                max_depth=_env_int("CRAWLER_MAX_DEPTH", 2),
                include_external=_env_bool("CRAWLER_INCLUDE_EXTERNAL", False),
                verbose=_env_bool("CRAWLER_VERBOSE", False),
            ),
            chunker=ChunkerConfig(
                chunk_size=_env_int("CHUNK_SIZE", 500),
                chunk_overlap=_env_int("CHUNK_OVERLAP", 50),
            ),
            embedder=EmbedderConfig(
                model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                batch_size=_env_int("EMBEDDING_BATCH_SIZE", 32),
                normalize_embeddings=_env_bool("EMBEDDING_NORMALIZE", True),
            ),
            qdrant=QdrantConfig(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv(
                    "QDRANT_COLLECTION", "website_chunks"
                ),
                collection_prefix=os.getenv("QDRANT_COLLECTION_PREFIX", ""),
                vector_size=(
                    int(os.environ["QDRANT_VECTOR_SIZE"])
                    if os.getenv("QDRANT_VECTOR_SIZE")
                    else None
                ),
                recreate_collection=_env_bool("QDRANT_RECREATE_COLLECTION", False),
            ),
            chat=ChatConfig(
                llm_model=os.getenv("LLM_MODEL", "google_genai:gemini-3.5-flash-lite"),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
                top_k=_env_int("RETRIEVAL_TOP_K", 5),
                cors_origins=[
                    origin.strip()
                    for origin in os.getenv(
                        "CORS_ORIGINS", "http://localhost:5173"
                    ).split(",")
                    if origin.strip()
                ],
            ),
        )
