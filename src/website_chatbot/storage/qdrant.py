from __future__ import annotations

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from website_chatbot.config import QdrantConfig
from website_chatbot.models import EmbeddedChunk, RetrievedChunk
from website_chatbot.storage.base import VectorStore

_UPSERT_BATCH_SIZE = 64


class QdrantVectorStore(VectorStore):
    """Persist embedded chunks in a Qdrant collection."""

    def __init__(self, config: QdrantConfig | None = None) -> None:
        self.config = config or QdrantConfig()
        self._client: AsyncQdrantClient | None = None

    async def connect(self) -> None:
        kwargs: dict = {"url": self.config.url}
        if self.config.api_key:
            kwargs["api_key"] = self.config.api_key
        self._client = AsyncQdrantClient(**kwargs)

    async def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0
        if self._client is None:
            raise RuntimeError("Call connect() before upsert()")

        vector_size = self.config.vector_size or len(chunks[0].vector)
        await self._ensure_collection(vector_size)

        points = [
            PointStruct(
                id=item.chunk.chunk_id,
                vector=item.vector,
                payload=self._build_payload(item),
            )
            for item in chunks
        ]

        written = 0
        for start in range(0, len(points), _UPSERT_BATCH_SIZE):
            batch = points[start : start + _UPSERT_BATCH_SIZE]
            await self._client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
                wait=True,
            )
            written += len(batch)

        return written

    async def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        if self._client is None:
            raise RuntimeError("Call connect() before search()")

        response = await self._client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                text=(point.payload or {}).get("text", ""),
                source_url=(point.payload or {}).get("source_url", ""),
                title=(point.payload or {}).get("title", ""),
                score=point.score,
            )
            for point in response.points
        ]

    async def list_collections(self) -> list[str]:
        if self._client is None:
            raise RuntimeError("Call connect() before list_collections()")

        response = await self._client.get_collections()
        return sorted(collection.name for collection in response.collections)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
        self._client = None

    async def _ensure_collection(self, vector_size: int) -> None:
        assert self._client is not None
        name = self.config.collection_name

        if self.config.recreate_collection:
            await self._client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            return

        exists = await self._client.collection_exists(name)
        if exists:
            return

        await self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

    @staticmethod
    def _build_payload(item: EmbeddedChunk) -> dict:
        payload = item.to_payload()
        payload["model_name"] = item.model_name
        return payload
