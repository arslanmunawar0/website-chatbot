from __future__ import annotations

from abc import ABC, abstractmethod

from website_chatbot.models import Document, TextChunk


class BaseChunker(ABC):
    """Split documents into smaller text chunks."""

    @abstractmethod
    def chunk_documents(
        self,
        documents: list[Document],
        *,
        website_name: str | None = None,
    ) -> list[TextChunk]:
        """Return ordered chunks for all documents."""
