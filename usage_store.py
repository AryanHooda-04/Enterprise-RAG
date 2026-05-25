from __future__ import annotations

import csv
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from config import Settings, settings


def record_usage(
    operation: str,
    model: str,
    *,
    active_settings: Settings = settings,
    document_name: str | None = None,
    file_hash: str | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    input_count: int | None = None,
    output_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_settings.ensure_directories()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "model": model,
        "document_name": document_name,
        "file_hash": file_hash,
        "prompt_tokens": prompt_tokens or 0,
        "completion_tokens": completion_tokens or 0,
        "total_tokens": total_tokens or (prompt_tokens or 0) + (completion_tokens or 0),
        "input_count": input_count or 0,
        "output_count": output_count or 0,
        "metadata": metadata or {},
    }
    with active_settings.usage_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def record_openai_usage(
    operation: str,
    model: str,
    usage: Any,
    *,
    active_settings: Settings = settings,
    document_name: str | None = None,
    file_hash: str | None = None,
    input_count: int | None = None,
    output_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return record_usage(
        operation,
        model,
        active_settings=active_settings,
        document_name=document_name,
        file_hash=file_hash,
        prompt_tokens=_usage_value(usage, "prompt_tokens", "input_tokens"),
        completion_tokens=_usage_value(usage, "completion_tokens", "output_tokens"),
        total_tokens=_usage_value(usage, "total_tokens"),
        input_count=input_count,
        output_count=output_count,
        metadata=metadata,
    )


def load_usage(active_settings: Settings = settings) -> list[dict[str, Any]]:
    if not active_settings.usage_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with active_settings.usage_path.open("r", encoding="utf-8") as file:
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


def summarize_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_operation: Counter[str] = Counter()
    by_model: Counter[str] = Counter()
    tokens_by_operation: defaultdict[str, int] = defaultdict(int)
    docs: set[str] = set()

    for record in records:
        operation = record.get("operation") or "unknown"
        model = record.get("model") or "unknown"
        by_operation[operation] += 1
        by_model[model] += 1
        tokens_by_operation[operation] += int(record.get("total_tokens") or 0)
        if record.get("document_name"):
            docs.add(record["document_name"])

    return {
        "calls": len(records),
        "tokens": sum(int(record.get("total_tokens") or 0) for record in records),
        "documents": len(docs),
        "by_operation": dict(by_operation),
        "by_model": dict(by_model),
        "tokens_by_operation": dict(tokens_by_operation),
    }


def usage_csv(records: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    fieldnames = [
        "timestamp",
        "operation",
        "model",
        "document_name",
        "file_hash",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_count",
        "output_count",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        writer.writerow({field: record.get(field, "") for field in fieldnames})
    return output.getvalue()


def usage_jsonl(records: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def _usage_value(usage: Any, *names: str) -> int:
    if not usage:
        return 0
    for name in names:
        value = getattr(usage, name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0
