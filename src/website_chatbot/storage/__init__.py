from website_chatbot.storage.base import VectorStore

__all__ = ["VectorStore", "QdrantVectorStore"]


def __getattr__(name: str):
    if name == "QdrantVectorStore":
        from website_chatbot.storage.qdrant import QdrantVectorStore

        return QdrantVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
