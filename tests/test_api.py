from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from website_chatbot.api import create_app
from website_chatbot.config import Settings


@pytest.fixture
def client(embedder, store, llm):
    app = create_app(settings=Settings(), embedder=embedder, vector_store=store, llm=llm)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_lists_sites(client):
    assert client.get("/api/sites").json() == {"sites": ["empty_site", "shop"]}


def test_chat_returns_answer_with_sources(client):
    response = client.post(
        "/api/chat",
        json={
            "site": "shop",
            "question": "How fast do you ship?",
            "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Orders ship within 2 business days [1]."
    assert body["sources"][0] == {"number": 1, "title": "Shipping", "url": "https://shop.example/shipping"}


def test_chat_unknown_site_is_404(client):
    response = client.post("/api/chat", json={"site": "nope", "question": "Hi"})

    assert response.status_code == 404


def test_chat_rejects_empty_question(client):
    response = client.post("/api/chat", json={"site": "shop", "question": ""})

    assert response.status_code == 422


def test_vector_store_is_closed_on_shutdown(embedder, store, llm):
    app = create_app(settings=Settings(), embedder=embedder, vector_store=store, llm=llm)
    with TestClient(app):
        assert store.connected
    assert not store.connected
