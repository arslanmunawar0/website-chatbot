from __future__ import annotations

from website_chatbot.embedding.sentence_transformer import with_title
from website_chatbot.models import TextChunk


def _chunk(title: str) -> TextChunk:
    return TextChunk(
        chunk_id="1", text="Props are declared with defineProps().", source_url="https://x", title=title, chunk_index=3
    )


def test_chunks_are_embedded_with_their_page_title():
    assert with_title(_chunk("Props | Vue.js")) == "Props | Vue.js\n\nProps are declared with defineProps()."


def test_chunks_without_title_are_embedded_as_is():
    assert with_title(_chunk("")) == "Props are declared with defineProps()."
