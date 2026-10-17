from __future__ import annotations

import asyncio

from qdrant_client import AsyncQdrantClient

from website_chatbot.config import QdrantConfig
from website_chatbot.models import EmbeddedChunk, TextChunk
from website_chatbot.storage.qdrant import QdrantVectorStore


def _chunk(chunk_id: str, text: str, url: str, vector: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=TextChunk(
            chunk_id=chunk_id,
            text=text,
            source_url=url,
            title=text.split()[0],
            chunk_index=0,
            metadata={"website": "shop"},
        ),
        vector=vector,
        model_name="fake",
    )


def test_upsert_then_search_returns_nearest_chunks_first():
    async def scenario():
        store = QdrantVectorStore(QdrantConfig(collection_name="shop"))
        store._client = AsyncQdrantClient(location=":memory:")
        await store.upsert(
            [
                _chunk("00000000-0000-0000-0000-000000000001", "Shipping takes 2 days", "https://s/ship", [1.0, 0.0]),
                _chunk("00000000-0000-0000-0000-000000000002", "Returns within 30 days", "https://s/returns", [0.0, 1.0]),
            ]
        )

        results = await store.search("shop", [0.9, 0.1], limit=2)
        collections = await store.list_collections()
        await store.close()
        return results, collections

    results, collections = asyncio.run(scenario())

    assert collections == ["shop"]
    assert [r.source_url for r in results] == ["https://s/ship", "https://s/returns"]
    assert results[0].text == "Shipping takes 2 days"
    assert results[0].title == "Shipping"
    assert results[0].score > results[1].score
