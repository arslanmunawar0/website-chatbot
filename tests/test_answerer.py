from __future__ import annotations

import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from website_chatbot.qa import ChatTurn, Source, WebsiteAnswerer
from website_chatbot.qa.answerer import NO_CONTEXT_ANSWER, _number_sources

from conftest import SHIPPING_CHUNKS


def test_answer_returns_llm_text_and_deduplicated_sources(embedder, store, llm):
    answerer = WebsiteAnswerer(embedder, store, llm, top_k=3)

    answer = asyncio.run(answerer.answer("shop", "How fast do you ship?"))

    assert answer.text == "Orders ship within 2 business days [1]."
    assert answer.sources == [
        Source(number=1, title="Shipping", url="https://shop.example/shipping"),
        Source(number=2, title="Returns", url="https://shop.example/returns"),
    ]
    assert embedder.queries == ["How fast do you ship?"]
    assert store.searches == [("shop", [0.1, 0.2, 0.3], 3)]


def test_answer_without_context_skips_llm(embedder, store):
    llm = FakeListChatModel(responses=[])
    answerer = WebsiteAnswerer(embedder, store, llm)

    answer = asyncio.run(answerer.answer("empty_site", "Anything?"))

    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.sources == []


def test_history_and_context_reach_the_prompt(embedder, store):
    captured = {}

    class RecordingLLM(FakeListChatModel):
        def _call(self, messages, *args, **kwargs):
            captured["messages"] = messages
            return super()._call(messages, *args, **kwargs)

    answerer = WebsiteAnswerer(embedder, store, RecordingLLM(responses=["ok"]))
    history = [
        ChatTurn(role="user", content="Hi"),
        ChatTurn(role="assistant", content="Hello! How can I help?"),
    ]

    asyncio.run(answerer.answer("shop", "Do you ship to Germany?", history))

    system, *rest = captured["messages"]
    assert "Orders ship within 2 business days." in system.content
    assert "[2] Returns" in system.content
    assert [m.content for m in rest] == ["Hi", "Hello! How can I help?", "Do you ship to Germany?"]


def test_number_sources_groups_chunks_by_page():
    sources, context = _number_sources(SHIPPING_CHUNKS)

    assert [s.number for s in sources] == [1, 2]
    assert context.count("[1] Shipping") == 2
    assert context.count("[2] Returns") == 1
