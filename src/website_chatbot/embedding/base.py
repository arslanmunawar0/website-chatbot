from __future__ import annotations

from abc import ABC, abstractmethod

from website_chatbot.models import EmbeddedChunk, TextChunk


class BaseEmbedder(ABC):
    """Turn text chunks into dense vectors."""

    @abstractmethod
    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        """Embed all chunks and return vectors aligned with input order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query into the same vector space as the chunks."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Identifier of the embedding model (for storage metadata)."""

    @property
    @abstractmethod
    def vector_dimensions(self) -> int:
        """Length of each embedding vector."""
