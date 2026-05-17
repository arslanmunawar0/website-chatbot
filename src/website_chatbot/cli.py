from __future__ import annotations

import argparse
import asyncio

from website_chatbot.runner import IngestionOptions, run_ingestion


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crawl websites, chunk text, and create embeddings."
    )
    parser.add_argument("--urls", nargs="+", help="One or more seed URLs to crawl")
    parser.add_argument(
        "--websites-file",
        help="websites.json: site name → parent + child URLs",
    )
    parser.add_argument(
        "--urls-file",
        help="(deprecated) use --websites-file with websites.json",
    )
    parser.add_argument(
        "--site",
        help="Website name for Qdrant collection (default: from URL hostname)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Write JSON output (multi-site: one file per site)",
    )
    parser.add_argument(
        "--deep-crawl",
        action="store_true",
        help="Follow internal links with BFS (uses CRAWLER_MAX_DEPTH)",
    )
    parser.add_argument("--css-selector", help="CSS selector for page content")
    parser.add_argument("--url-contains", help="Keep URLs containing this substring")
    parser.add_argument("--model", help="Sentence-transformers model name")
    parser.add_argument("--chunk-size", type=int, help="Characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap")
    parser.add_argument(
        "--qdrant",
        action="store_true",
        help="Upsert into Qdrant (one collection per website)",
    )
    parser.add_argument(
        "--qdrant-recreate",
        action="store_true",
        help="Recreate each site's Qdrant collection before upsert",
    )
    parser.add_argument("--verbose", action="store_true", help="Print crawl progress")
    return parser


def _options_from_args(args: argparse.Namespace) -> IngestionOptions:
    return IngestionOptions(
        urls=args.urls,
        websites_file=args.websites_file,
        urls_file=args.urls_file,
        site=args.site,
        output=args.output,
        deep_crawl=True if args.deep_crawl else None,
        css_selector=args.css_selector,
        url_contains=args.url_contains,
        model=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        qdrant=args.qdrant,
        qdrant_recreate=True if args.qdrant_recreate else None,
        verbose=True if args.verbose else None,
    )


async def _async_main(args: argparse.Namespace) -> int:
    if not args.urls and not args.websites_file and not args.urls_file:
        print("Error: provide --urls or --websites-file")
        return 1
    return await run_ingestion(_options_from_args(args))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
