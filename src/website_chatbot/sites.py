from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

_SITE_URL_RE = re.compile(r"^([^|]+)\|(.+)$")
_SECTION_RE = re.compile(r"^\[(.+)\]\s*$")


@dataclass(frozen=True, slots=True)
class SiteIngestion:
    """One website and its seed URLs — maps to one Qdrant collection."""

    website_name: str
    urls: list[str]

    @property
    def collection_name(self) -> str:
        return collection_name_for_website(self.website_name)


def sanitize_website_name(name: str) -> str:
    """Turn a display name or hostname into a safe collection identifier."""
    cleaned = name.strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError(f"Invalid website name: {name!r}")
    return cleaned[:128]


def collection_name_for_website(
    website_name: str,
    *,
    prefix: str = "",
) -> str:
    base = sanitize_website_name(website_name)
    if not prefix:
        return base
    prefix_clean = sanitize_website_name(prefix) if prefix else ""
    return f"{prefix_clean}_{base}" if prefix_clean else base


def website_name_from_url(url: str) -> str:
    """Derive a website id from a URL hostname (e.g. help.bbc.com → help_bbc_com)."""
    host = urlparse(url.strip()).hostname or "unknown"
    return sanitize_website_name(host.replace(".", "_"))


def _urls_from_site_entry(entry: Any) -> list[str]:
    """Normalize a JSON site entry into a flat URL list (parent + children)."""
    if isinstance(entry, str):
        return [entry.strip()]

    if isinstance(entry, list):
        return [str(u).strip() for u in entry if str(u).strip()]

    if isinstance(entry, dict):
        urls: list[str] = []
        parent = entry.get("parent") or entry.get("seed") or entry.get("root")
        if parent:
            urls.append(str(parent).strip())

        children = entry.get("children") or entry.get("links") or entry.get("urls")
        if children:
            if isinstance(children, str):
                urls.append(children.strip())
            else:
                urls.extend(str(u).strip() for u in children if str(u).strip())

        if urls:
            return urls

        raise ValueError(
            "Each website entry needs 'parent' + 'children', or a 'urls' list"
        )

    raise ValueError(f"Unsupported site entry type: {type(entry).__name__}")


def parse_websites_json(path: Path) -> dict[str, list[str]]:
    """
    Parse websites.json. Supported shapes:

    .. code-block:: json

        {
          "bbc": {
            "parent": "https://example.com",
            "children": ["https://example.com/a", "https://example.com/b"]
          },
          "mysite": ["https://example.com", "https://example.com/about"]
        }

    Also accepts a top-level ``"websites"`` key wrapping the same object.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and "websites" in data:
        data = data["websites"]

    if not isinstance(data, dict):
        raise ValueError("websites.json must be a JSON object keyed by website name")

    grouped: dict[str, list[str]] = {}
    for raw_name, entry in data.items():
        if str(raw_name).startswith("_"):
            continue
        site = sanitize_website_name(str(raw_name))
        grouped[site] = _dedupe_urls(_urls_from_site_entry(entry))

    return grouped


def parse_urls_file(path: Path) -> dict[str, list[str]]:
    """
    Parse a urls file. Supported line formats:

    - ``site_name|https://...``  explicit website
    - ``[site_name]``            section header for following URLs
    - ``https://...``            auto-grouped by URL hostname
    """
    grouped: dict[str, list[str]] = {}
    current_site: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        section = _SECTION_RE.match(line)
        if section:
            current_site = sanitize_website_name(section.group(1))
            grouped.setdefault(current_site, [])
            continue

        explicit = _SITE_URL_RE.match(line)
        if explicit:
            site = sanitize_website_name(explicit.group(1))
            url = explicit.group(2).strip()
            grouped.setdefault(site, []).append(url)
            current_site = site
            continue

        site = current_site or website_name_from_url(line)
        grouped.setdefault(site, []).append(line)

    return grouped


def build_site_ingestions(
    *,
    cli_urls: list[str] | None = None,
    websites_file: str | Path | None = None,
    urls_file: str | Path | None = None,
    site_override: str | None = None,
) -> list[SiteIngestion]:
    """
    Build per-website ingestion jobs from CLI args and/or websites.json.

    - ``--site bbc --urls ...`` → all URLs under website ``bbc``
    - JSON file with multiple sites → one job per site
    - plain URLs without ``--site`` → grouped by hostname
    """
    grouped: dict[str, list[str]] = {}

    config_path = websites_file or urls_file
    if config_path:
        path = Path(config_path)
        if path.suffix.lower() == ".json":
            grouped.update(parse_websites_json(path))
        else:
            grouped.update(parse_urls_file(path))

    for url in cli_urls or []:
        site = (
            sanitize_website_name(site_override)
            if site_override
            else website_name_from_url(url)
        )
        grouped.setdefault(site, []).append(url.strip())

    if not grouped:
        return []

    if site_override and len(grouped) > 1:
        merged: dict[str, list[str]] = {
            sanitize_website_name(site_override): []
        }
        for urls in grouped.values():
            merged[sanitize_website_name(site_override)].extend(urls)
        grouped = merged

    return [
        SiteIngestion(website_name=site, urls=_dedupe_urls(urls))
        for site, urls in sorted(grouped.items())
    ]


def _dedupe_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            unique.append(url)
    return unique
