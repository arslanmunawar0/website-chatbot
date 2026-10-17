from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from website_chatbot.embedding.base import BaseEmbedder
from website_chatbot.models import EmbeddedChunk, RetrievedChunk, TextChunk
from website_chatbot.storage.base import VectorStore


class FakeEmbedder(BaseEmbedder):
    model_name = "fake"
    vector_dimensions = 3

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        return [EmbeddedChunk(chunk=c, vector=[0.0, 0.0, 0.0], model_name="fake") for c in chunks]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorStore(VectorStore):
    def __init__(self, collections: dict[str, list[RetrievedChunk]]) -> None:
        self.collections = collections
        self.searches: list[tuple[str, list[float], int]] = []
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        return len(chunks)

    async def search(self, collection: str, vector: list[float], limit: int) -> list[RetrievedChunk]:
        self.searches.append((collection, vector, limit))
        return self.collections.get(collection, [])[:limit]

    async def list_collections(self) -> list[str]:
        return sorted(self.collections)

    async def close(self) -> None:
        self.connected = False


SHIPPING_CHUNKS = [
    RetrievedChunk(
        text="Orders ship within 2 business days.",
        source_url="https://shop.example/shipping",
        title="Shipping",
        score=0.91,
    ),
    RetrievedChunk(
        text="Express delivery is available in the EU.",
        source_url="https://shop.example/shipping",
        title="Shipping",
        score=0.85,
    ),
    RetrievedChunk(
        text="Returns are accepted within 30 days.",
        source_url="https://shop.example/returns",
        title="Returns",
        score=0.72,
    ),
]


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def store() -> FakeVectorStore:
    return FakeVectorStore({"shop": SHIPPING_CHUNKS, "empty_site": []})


@pytest.fixture
def llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["Orders ship within 2 business days [1]."])
