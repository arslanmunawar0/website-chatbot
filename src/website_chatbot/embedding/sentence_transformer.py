from __future__ import annotations

from sentence_transformers import SentenceTransformer

from website_chatbot.config import EmbedderConfig
from website_chatbot.embedding.base import BaseEmbedder
from website_chatbot.models import EmbeddedChunk, TextChunk


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings via sentence-transformers."""

    def __init__(self, config: EmbedderConfig | None = None) -> None:
        self.config = config or EmbedderConfig()
        self._model = SentenceTransformer(self.config.model_name)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    @property
    def vector_dimensions(self) -> int:
        get_dimension = getattr(
            self._model,
            "get_embedding_dimension",
            self._model.get_sentence_embedding_dimension,
        )
        return int(get_dimension())

    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            return []

        texts = [with_title(chunk) for chunk in chunks]
        vectors = self._model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=len(texts) > 32,
        )

        return [
            EmbeddedChunk(
                chunk=chunk,
                vector=vector.tolist(),
                model_name=self.model_name,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            text,
            normalize_embeddings=self.config.normalize_embeddings,
        )
        return vector.tolist()


def with_title(chunk: TextChunk) -> str:
    """Prefix the page title so chunks without their own heading still embed with their topic."""
    return f"{chunk.title}\n\n{chunk.text}" if chunk.title else chunk.text
