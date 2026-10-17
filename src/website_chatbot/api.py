"""
HTTP API for chatting with ingested websites.

  GET  /api/health   liveness check
  GET  /api/sites    websites (Qdrant collections) available to chat with
  POST /api/chat     ask a question about one website

Run with:  chatbot-api   (or: uvicorn website_chatbot.api:create_app --factory --reload)
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from website_chatbot.config import Settings
from website_chatbot.embedding.base import BaseEmbedder
from website_chatbot.qa import ChatTurn, WebsiteAnswerer
from website_chatbot.storage.base import VectorStore

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    site: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=2000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class SourceOut(BaseModel):
    number: int
    title: str
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


class SitesResponse(BaseModel):
    sites: list[str]


def create_app(
    *,
    settings: Settings | None = None,
    embedder: BaseEmbedder | None = None,
    vector_store: VectorStore | None = None,
    llm: BaseChatModel | None = None,
) -> FastAPI:
    """Build the API. Dependencies left as ``None`` are created from ``.env`` at startup."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = vector_store or _default_vector_store(settings)
        await store.connect()
        app.state.vector_store = store
        app.state.answerer = WebsiteAnswerer(
            embedder or _default_embedder(settings),
            store,
            llm or _default_llm(settings),
            top_k=settings.chat.top_k,
        )
        try:
            yield
        finally:
            await store.close()

    app = FastAPI(title="Website Chatbot API", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.chat.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/sites", response_model=SitesResponse)
    async def list_sites(request: Request) -> SitesResponse:
        sites = await request.app.state.vector_store.list_collections()
        return SitesResponse(sites=sites)

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        sites = await request.app.state.vector_store.list_collections()
        if body.site not in sites:
            raise HTTPException(status_code=404, detail=f"Unknown site: {body.site}")

        answer = await request.app.state.answerer.answer(
            body.site,
            body.question,
            [ChatTurn(role=m.role, content=m.content) for m in body.history],
        )
        return ChatResponse(
            answer=answer.text,
            sources=[
                SourceOut(number=s.number, title=s.title, url=s.url)
                for s in answer.sources
            ],
        )

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def _default_vector_store(settings: Settings) -> VectorStore:
    from website_chatbot.storage.qdrant import QdrantVectorStore

    return QdrantVectorStore(settings.qdrant)


def _default_embedder(settings: Settings) -> BaseEmbedder:
    from website_chatbot.embedding.sentence_transformer import (
        SentenceTransformerEmbedder,
    )

    return SentenceTransformerEmbedder(settings.embedder)


def _default_llm(settings: Settings) -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.chat.llm_model, temperature=settings.chat.temperature
    )


def run() -> None:
    import os

    import uvicorn

    uvicorn.run(
        "website_chatbot.api:create_app",
        factory=True,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )
