from __future__ import annotations

from website_chatbot.chunking.base import BaseChunker
from website_chatbot.crawler.base import BaseCrawler
from website_chatbot.embedding.base import BaseEmbedder
from website_chatbot.models import Document, EmbeddedChunk, TextChunk
from website_chatbot.storage.base import VectorStore


class IngestionPipeline:
    """
    Orchestrates: crawl URLs → chunk documents → embed chunks → optional store.
    """

    def __init__(
        self,
        crawler: BaseCrawler,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_store: VectorStore | None = None,
    ) -> None:
        self.crawler = crawler
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    async def run(
        self,
        urls: list[str],
        *,
        website_name: str | None = None,
    ) -> list[EmbeddedChunk]:
        documents = await self.crawl(urls)
        chunks = self.chunk(documents, website_name=website_name)
        return self.embed(chunks)

    async def crawl(self, urls: list[str]) -> list[Document]:
        return await self.crawler.crawl(urls)

    def chunk(
        self,
        documents: list[Document],
        *,
        website_name: str | None = None,
    ) -> list[TextChunk]:
        return self.chunker.chunk_documents(
            documents, website_name=website_name
        )

    def embed(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        embedded = self.embedder.embed(chunks)
        return embedded

    async def run_and_store(
        self,
        urls: list[str],
        *,
        website_name: str | None = None,
    ) -> list[EmbeddedChunk]:
        embedded = await self.run(urls, website_name=website_name)
        if self.vector_store is not None:
            async with self.vector_store:
                await self.vector_store.upsert(embedded)
        return embedded
