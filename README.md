# Website Chatbot — Ingestion Pipeline

Crawl websites → split into chunks → embed with sentence-transformers → store in Qdrant.

## Project layout

```
src/website_chatbot/
├── config.py          # Settings (env vars)
├── models.py          # Document, TextChunk, EmbeddedChunk
├── pipeline.py        # Orchestrates crawl → chunk → embed
├── crawler/           # Crawl4AI implementation (+ base interface)
├── chunking/          # Recursive text splitter (+ base interface)
├── embedding/         # Sentence-transformers (+ base interface)
└── storage/           # VectorStore + QdrantVectorStore
scripts/ingest.py      # CLI
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```

Start Qdrant locally (Docker):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Usage

### Easiest: `python main.py`

1. Copy `websites.example.json` → `websites.json` and add your sites.
2. Configure `.env` (Qdrant URL, API key, etc.).
3. Run:

```bash
python main.py
```

This runs **crawl → chunk → embed → Qdrant** automatically (reads `websites.json` + `.env`).

**`websites.json` format** (website name → parent + child links):

```json
{
  "bbc": {
    "parent": "https://help.bbc.com/hc/en-us",
    "children": [
      "https://help.bbc.com/hc/en-us/articles/getting-started",
      "https://help.bbc.com/hc/en-us/articles/account-help"
    ]
  },
  "example": {
    "parent": "https://example.com",
    "children": ["https://example.com/about", "https://example.com/faq"]
  }
}
```

Each key becomes a Qdrant collection (`bbc`, `example`, …).

**Or use the `ingest` CLI** for one-off flags:

**Crawl, embed, and save to JSON:**

```bash
ingest --urls https://example.com --output data/embeddings.json --verbose
```

**Crawl, embed, and upsert into Qdrant** (collection name = website name, e.g. `example_com`):

```bash
ingest --urls https://example.com --qdrant --verbose
```

**Explicit website name** (collection `bbc`):

```bash
ingest --site bbc --urls https://help.bbc.com/hc/en-us --qdrant --verbose
```

**Multiple websites** from JSON:

```bash
ingest --websites-file websites.json --qdrant --verbose
```

**Deep crawl + Qdrant (replace collection first):**

```bash
ingest \
  --urls https://help.bbc.com/hc/en-us \
  --deep-crawl \
  --css-selector main \
  --url-contains /articles/ \
  --qdrant \
  --qdrant-recreate \
  --verbose
```

**Config:** copy `.env.example` to `.env`, add your real Qdrant URL and API key in `.env` only (never commit `.env`). The CLI loads `.env` automatically on startup.

## Extending

| Step | Swap by implementing |
|------|----------------------|
| Crawl | `BaseCrawler` |
| Chunk | `BaseChunker` |
| Embed | `BaseEmbedder` |
| Store | `VectorStore` |
