from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


def new_chunk_id() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True, slots=True)
class Document:
    """Raw page extracted by the crawler."""

    url: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A slice of a document, ready for embedding."""

    chunk_id: str
    text: str
    source_url: str
    title: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Chunk plus its dense vector."""

    chunk: TextChunk
    vector: list[float]
    model_name: str

    def to_payload(self) -> dict[str, Any]:
        """Metadata stored alongside the vector in Qdrant."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "text": self.chunk.text,
            "source_url": self.chunk.source_url,
            "title": self.chunk.title,
            "chunk_index": self.chunk.chunk_index,
            "website": self.chunk.metadata.get("website", ""),
            **{
                k: v
                for k, v in self.chunk.metadata.items()
                if k not in {"website"}
            },
        }


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A stored chunk returned by a similarity search."""

    text: str
    source_url: str
    title: str
    score: float
