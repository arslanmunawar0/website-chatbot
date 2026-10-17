from website_chatbot.embedding.base import BaseEmbedder

__all__ = ["BaseEmbedder", "SentenceTransformerEmbedder"]


def __getattr__(name: str):
    if name == "SentenceTransformerEmbedder":
        from website_chatbot.embedding.sentence_transformer import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
