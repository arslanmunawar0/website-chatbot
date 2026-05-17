#!/usr/bin/env python3
"""
Run the full pipeline from websites.json + .env:

  crawl → chunk → embed → save to Qdrant Cloud

Usage:
  python main.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from website_chatbot.runner import IngestionOptions, run_ingestion

PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_websites_file() -> Path:
    name = os.getenv("WEBSITES_FILE", "websites.json")
    path = Path(name)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main() -> None:
    config_path = _resolve_websites_file()

    if not config_path.exists():
        example = PROJECT_ROOT / "websites.example.json"
        print(f"Missing {config_path}")
        if example.exists():
            print("  Copy the template:  cp websites.example.json websites.json")
            print("  Add your websites (parent + child links), then run again.")
        sys.exit(1)

    print(f"Reading websites from: {config_path}")
    print("Settings from: .env\n")

    options = IngestionOptions.from_env_defaults(config_path)
    raise SystemExit(asyncio.run(run_ingestion(options)))


if __name__ == "__main__":
    main()
