#!/usr/bin/env python3
"""Incrementally append current Danbooru character tags to the local CSV.

The existing CSV contains valuable Chinese names and aliases, so this updater
never rewrites or deletes old rows. It fetches official, non-deprecated
category=4 tags and appends only names that are not already present.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

API_URL = "https://danbooru.donmai.us/tags.json"
PAGE_SIZE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "danbooru角色标签.csv",
    )
    parser.add_argument("--min-post-count", type=int, default=25)
    parser.add_argument("--max-pages", type=int, default=1000)
    parser.add_argument("--request-delay", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_existing_names(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if row and row[0].strip():
                names.add(row[0].strip())
    return names


def fetch_page(
    client: httpx.Client,
    page: int,
    min_post_count: int,
) -> list[dict]:
    params = {
        "search[category]": 4,
        "search[is_deprecated]": "false",
        "search[post_count]": f">={min_post_count}",
        "search[order]": "count",
        "limit": PAGE_SIZE,
        "page": page,
    }
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            response = client.get(API_URL, params=params)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, list):
                raise RuntimeError("Danbooru API returned a non-list response")
            return value
        except (httpx.HTTPError, ValueError, RuntimeError) as error:
            last_error = error
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch page {page}: {last_error}")


def humanize_tag(name: str) -> str:
    return name.replace("_", " ").strip()


def main() -> int:
    args = parse_args()
    if args.min_post_count < 1:
        raise SystemExit("--min-post-count must be at least 1")
    existing = load_existing_names(args.csv)
    additions: dict[str, dict] = {}
    fetched = 0

    with httpx.Client(
        timeout=httpx.Timeout(20.0, connect=10.0),
        follow_redirects=True,
        # AstrBot deployments commonly expose their outbound proxy through
        # standard environment variables; honor it just like curl/git do.
        trust_env=True,
        headers={"User-Agent": "AstrBot-NovelAI-character-sync/1.0"},
    ) as client:
        for page in range(1, args.max_pages + 1):
            rows = fetch_page(client, page, args.min_post_count)
            if not rows:
                break
            fetched += len(rows)
            for item in rows:
                name = str(item.get("name", "") or "").strip()
                if not name or name in existing:
                    continue
                additions[name] = item
            if page % 25 == 0:
                print(
                    json.dumps(
                        {
                            "progress_pages": page,
                            "fetched": fetched,
                            "pending_additions": len(additions),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if len(rows) < PAGE_SIZE:
                break
            if int(rows[-1].get("post_count", 0) or 0) < args.min_post_count:
                break
            if args.request_delay > 0:
                time.sleep(args.request_delay)

    ordered = sorted(
        additions.values(),
        key=lambda item: int(item.get("post_count", 0) or 0),
        reverse=True,
    )
    summary = {
        "source": API_URL,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "min_post_count": args.min_post_count,
        "fetched": fetched,
        "existing": len(existing),
        "added": len(ordered),
        "dry_run": args.dry_run,
    }
    if not args.dry_run and ordered:
        with args.csv.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            for item in ordered:
                name = str(item["name"])
                writer.writerow([
                    name,
                    4,
                    int(item.get("post_count", 0) or 0),
                    humanize_tag(name),
                ])
        metadata_path = args.csv.parent / "data" / "character_database_sync.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
