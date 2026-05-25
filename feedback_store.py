from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

from config import Settings, settings


def save_feedback(payload: dict[str, Any], active_settings: Settings = settings) -> dict[str, Any]:
    active_settings.ensure_directories()
    record = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with active_settings.feedback_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_feedback(active_settings: Settings = settings) -> list[dict[str, Any]]:
    if not active_settings.feedback_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with active_settings.feedback_path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def feedback_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def feedback_csv(records: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = [
        "submitted_at",
        "sentiment",
        "bad_retrieval",
        "comment",
        "query",
        "answer",
        "role",
        "username",
        "model",
        "embedding_model",
        "search_mode",
        "confidence",
        "source_count",
        "top_sources",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for record in records:
        sources = record.get("source_metadata") or []
        top_sources = "; ".join(
            f"{source.get('file_name', 'unknown')}#{source.get('chunk_index', 'n/a')}:{source.get('score', '')}"
            for source in sources[:5]
            if isinstance(source, dict)
        )
        writer.writerow(
            {
                "submitted_at": record.get("submitted_at", ""),
                "sentiment": record.get("sentiment", ""),
                "bad_retrieval": record.get("bad_retrieval", ""),
                "comment": record.get("comment", ""),
                "query": record.get("query", ""),
                "answer": record.get("answer", ""),
                "role": record.get("role", ""),
                "username": record.get("username", ""),
                "model": record.get("model", ""),
                "embedding_model": record.get("embedding_model", ""),
                "search_mode": record.get("search_mode", ""),
                "confidence": record.get("confidence", ""),
                "source_count": record.get("source_count", ""),
                "top_sources": top_sources,
            }
        )
    return output.getvalue()
