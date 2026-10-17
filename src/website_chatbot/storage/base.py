from __future__ import annotations

from abc import ABC, abstractmethod

from website_chatbot.models import EmbeddedChunk, RetrievedChunk


class VectorStore(ABC):
    """Persist embedded chunks (Qdrant, Pinecone, etc.)."""

    @abstractmethod
    async def connect(self) -> None:
        """Open client connections."""

    @abstractmethod
    async def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        """Insert or update chunks. Returns number of objects written."""

    @abstractmethod
    async def search(
        self, collection: str, vector: list[float], limit: int
    ) -> list[RetrievedChunk]:
        """Return the chunks in ``collection`` most similar to ``vector``."""

    @abstractmethod
    async def list_collections(self) -> list[str]:
        """Names of all collections (one per ingested website)."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    async def __aenter__(self) -> VectorStore:
        await self.connect()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
