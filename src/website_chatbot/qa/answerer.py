from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from website_chatbot.embedding.base import BaseEmbedder
from website_chatbot.models import RetrievedChunk
from website_chatbot.storage.base import VectorStore

NO_CONTEXT_ANSWER = (
    "I couldn't find anything about that on this website. "
    "Try rephrasing your question, or contact the website's team directly."
)

_SYSTEM_PROMPT = """You are a friendly support assistant for the website "{website}".

Answer the user's question using ONLY the numbered excerpts below, which were taken from the website's pages.
- Answer directly and naturally, as the website's own assistant. Don't mention "excerpts", "context" or "the provided documentation".
- If the excerpts answer the question only partly, answer what they cover without adding a disclaimer.
- Only if nothing relevant is in the excerpts, say you couldn't find that on this website. Never invent facts.
- Be concise: short paragraphs or bullet points, and short code snippets when they help.
- Cite the pages you used with their numbers in square brackets, e.g. [1] or [2][3]. Cite only pages you actually used.

Context:
{context}"""

_CITATION = re.compile(r"\[(\d+)\]")

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)


@dataclass(frozen=True, slots=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class Source:
    number: int
    title: str
    url: str


@dataclass(frozen=True, slots=True)
class Answer:
    text: str
    sources: list[Source]


class WebsiteAnswerer:
    """
    Retrieval-augmented answering: embed question → search Qdrant → LLM answer with citations.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        llm: BaseChatModel,
        *,
        top_k: int = 8,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self._chain = _PROMPT | llm | StrOutputParser()

    async def answer(
        self,
        collection: str,
        question: str,
        history: list[ChatTurn] | None = None,
    ) -> Answer:
        vector = await asyncio.to_thread(self.embedder.embed_query, question)
        chunks = await self.vector_store.search(collection, vector, self.top_k)
        if not chunks:
            return Answer(text=NO_CONTEXT_ANSWER, sources=[])

        sources, context = _number_sources(chunks)
        text = await self._chain.ainvoke(
            {
                "website": collection,
                "context": context,
                "history": _to_messages(history or []),
                "question": question,
            }
        )
        text = text.strip()
        cited = {int(number) for number in _CITATION.findall(text)}
        return Answer(text=text, sources=[s for s in sources if s.number in cited])


def _number_sources(chunks: list[RetrievedChunk]) -> tuple[list[Source], str]:
    """Give each distinct page one citation number and build the prompt context."""
    sources: dict[str, Source] = {}
    blocks: list[str] = []
    for chunk in chunks:
        source = sources.get(chunk.source_url)
        if source is None:
            source = Source(
                number=len(sources) + 1,
                title=chunk.title or chunk.source_url,
                url=chunk.source_url,
            )
            sources[chunk.source_url] = source
        blocks.append(f"[{source.number}] {source.title}\n{chunk.text}")
    return list(sources.values()), "\n\n".join(blocks)


def _to_messages(history: list[ChatTurn]) -> list[BaseMessage]:
    return [
        HumanMessage(turn.content) if turn.role == "user" else AIMessage(turn.content)
        for turn in history
    ]
