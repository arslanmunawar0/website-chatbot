from __future__ import annotations

from website_chatbot.chunking.base import BaseChunker
from website_chatbot.config import ChunkerConfig
from website_chatbot.models import Document, TextChunk, new_chunk_id

_SEPARATORS = ("\n\n", "\n", ". ", " ")


class RecursiveTextChunker(BaseChunker):
    """Split text with overlap, preferring paragraph and sentence boundaries."""

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self.config = config or ChunkerConfig()
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def chunk_documents(
        self,
        documents: list[Document],
        *,
        website_name: str | None = None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        for doc in documents:
            pieces = self._split_text(doc.content)
            for index, text in enumerate(pieces):
                metadata = {
                    "source": doc.metadata.get("source", ""),
                    **{
                        k: v
                        for k, v in doc.metadata.items()
                        if k not in {"title", "description"}
                    },
                }
                if website_name:
                    metadata["website"] = website_name
                chunks.append(
                    TextChunk(
                        chunk_id=new_chunk_id(),
                        text=text,
                        source_url=doc.url,
                        title=doc.title,
                        chunk_index=index,
                        metadata=metadata,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        return self._recursive_split(text.strip(), _SEPARATORS)

    def _recursive_split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        if len(text) <= self.config.chunk_size:
            return [text] if text else []

        separator = separators[0] if separators else ""
        next_separators = separators[1:] if len(separators) > 1 else ("",)

        parts = text.split(separator) if separator else list(text)
        merged = self._merge_parts(parts, separator)

        chunks: list[str] = []
        current = ""

        for part in merged:
            candidate = f"{current}{part}" if current else part
            if len(candidate) <= self.config.chunk_size:
                current = candidate
                continue

            if current:
                chunks.extend(
                    self._recursive_split(current, next_separators)
                    if len(current) > self.config.chunk_size
                    else [current]
                )

            if len(part) > self.config.chunk_size and next_separators:
                chunks.extend(self._recursive_split(part, next_separators))
                current = ""
            else:
                current = part

        if current:
            chunks.extend(
                self._recursive_split(current, next_separators)
                if len(current) > self.config.chunk_size
                else [current]
            )

        return self._apply_overlap(chunks)

    def _merge_parts(self, parts: list[str], separator: str) -> list[str]:
        if not separator:
            return parts
        merged: list[str] = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                merged.append(part + separator)
            else:
                merged.append(part)
        return merged

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        if not chunks or self.config.chunk_overlap <= 0:
            return chunks

        overlapped: list[str] = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped.append(chunk)
                continue
            prev = overlapped[-1]
            tail = prev[-self.config.chunk_overlap :]
            overlapped.append(tail + chunk)
        return overlapped
