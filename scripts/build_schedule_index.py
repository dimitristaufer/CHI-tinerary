#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPO_ROOT / "chi_2026_schedule_export.csv"
OUT_JSON = REPO_ROOT / "chi_relevance_client" / "public" / "data" / "schedule_index.json"

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "llm", "llms", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves",
    "acm", "accepted", "affiliation", "al", "arxiv", "author", "authors", "conference", "copyright", "date", "department", "doi",
    "edition", "et", "figure", "figures", "http", "https", "institute", "isbn", "issn", "journal", "org", "page", "pages",
    "preprint", "proceedings", "section", "table", "tables", "university", "volume", "vol", "workshop", "www",
    "staufer", "morehouse", "hartmann", "berendt",
}
TOKEN_RE = re.compile(r"[a-z][a-z\-]{2,}")


def tokenize(text: str) -> list[str]:
    tokens = TOKEN_RE.findall((text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and not t.isdigit()]


def combine_schedule_text(row: dict[str, str]) -> str:
    parts = [
        row.get("title", ""),
        row.get("abstract", ""),
        row.get("session_titles", ""),
        row.get("session_type", ""),
        row.get("content_type", ""),
    ]
    return " ".join(p for p in parts if p)


def parse_optional_int(value: str | None) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def main() -> int:
    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    schedule_doc_freq: Counter[str] = Counter()
    packed_rows: list[dict[str, object]] = []

    for row in rows:
        counter = Counter(tokenize(combine_schedule_text(row)))
        schedule_doc_freq.update(counter.keys())

        packed_rows.append(
            {
                "title": row.get("title", ""),
                "authors": row.get("authors", ""),
                "abstract": row.get("abstract", ""),
                "room": row.get("room", ""),
                "building": row.get("building", ""),
                "day": row.get("day", ""),
                "session_type": row.get("session_type", ""),
                "start_date_unix_ms": parse_optional_int(row.get("start_date_unix_ms")),
                "end_date_unix_ms": parse_optional_int(row.get("end_date_unix_ms")),
                "tokens": [[term, count] for term, count in counter.items()],
            }
        )

    payload = {
        "version": 2,
        "row_count": len(packed_rows),
        "rows": packed_rows,
        "schedule_doc_freq": dict(schedule_doc_freq),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_mb = OUT_JSON.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_JSON} ({size_mb:.2f} MB, {len(packed_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
