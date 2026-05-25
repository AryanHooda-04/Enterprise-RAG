from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np

from config import Settings, settings
from errors import VectorStoreError


logger = logging.getLogger(__name__)


class VectorStore:
    """Local FAISS vector store using normalized vectors and inner product.

    Normalizing vectors before adding/searching turns IndexFlatIP into cosine
    similarity search.
    """

    def __init__(self, active_settings: Settings = settings):
        self.settings = active_settings
        self.index = None
        self.chunks: list[dict[str, Any]] = []
        self.documents: dict[str, dict[str, Any]] = {}
        self.load()

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0

    @property
    def total_vectors(self) -> int:
        if self.index is None:
            return 0
        return int(self.index.ntotal)

    def has_document(self, file_hash: str) -> bool:
        return file_hash in self.documents

    def get_document(self, file_hash: str) -> dict[str, Any] | None:
        return self.documents.get(file_hash)

    def list_documents(self) -> list[dict[str, Any]]:
        return list(self.documents.values())

    def load(self) -> None:
        self.settings.ensure_directories()

        if self.settings.index_path.exists():
            self.index = faiss.read_index(str(self.settings.index_path))
            logger.info("Loaded FAISS index from %s", self.settings.index_path)

        self.chunks = self._read_json_list(self.settings.chunks_path)
        self.documents = self._read_json_dict(self.settings.documents_path)

        if self.index is not None and self.index.ntotal != len(self.chunks):
            raise VectorStoreError(
                "FAISS index and chunk metadata are out of sync. "
                f"Index vectors={self.index.ntotal}, chunks={len(self.chunks)}."
            )

    def save(self) -> None:
        self.settings.ensure_directories()

        if self.index is not None:
            faiss.write_index(self.index, str(self.settings.index_path))
        elif self.settings.index_path.exists():
            self.settings.index_path.unlink()

        self._write_json(self.settings.chunks_path, self.chunks)
        self._write_json(self.settings.documents_path, self.documents)
        logger.info("Saved FAISS index and metadata to %s", self.settings.index_dir)

    def add_chunks(
        self,
        chunks: Iterable[dict[str, Any]],
        embeddings: list[list[float]],
        *,
        file_hash: str,
        file_name: str,
        source_path: str | None = None,
    ) -> int:
        chunk_list = list(chunks)
        if not chunk_list:
            raise VectorStoreError("No chunks were provided for indexing.")

        if len(chunk_list) != len(embeddings):
            raise VectorStoreError("Chunk count and embedding count do not match.")

        vectors = self._normalize_vectors(embeddings)
        self._ensure_index(vectors.shape[1])

        start_position = len(self.chunks)
        records: list[dict[str, Any]] = []

        for offset, chunk in enumerate(chunk_list):
            metadata = dict(chunk["metadata"])
            metadata["vector_position"] = start_position + offset
            metadata.setdefault("file_hash", file_hash)
            metadata.setdefault("file_name", file_name)
            metadata.setdefault("chunk_index", offset)
            metadata.setdefault("embedding_model", self.settings.openai_embedding_model)
            if source_path:
                metadata.setdefault("source_path", source_path)

            records.append(
                {
                    "id": metadata.get("chunk_id", f"{file_hash}:{offset}"),
                    "text": chunk["text"],
                    "metadata": metadata,
                }
            )

        self.index.add(vectors)
        self.chunks.extend(records)
        visual_chunk_count = sum(
            1 for record in records if record.get("metadata", {}).get("source_type") == "image"
        )
        self.documents[file_hash] = {
            "file_hash": file_hash,
            "file_name": file_name,
            "chunk_count": len(records),
            "visual_chunk_count": visual_chunk_count,
            "vector_start": start_position,
            "vector_end": start_position + len(records) - 1,
            "embedding_model": self.settings.openai_embedding_model,
            "source_path": source_path,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.info("Indexed %s chunks for %s", len(records), file_name)
        return len(records)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero.")

        if self.is_empty:
            return []

        query_vector = self._normalize_vectors([query_embedding])
        active_filters = _clean_filters(filters)
        k = self.total_vectors if active_filters else min(top_k, self.total_vectors)
        scores, indices = self.index.search(query_vector, k)

        results: list[dict[str, Any]] = []
        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue
            chunk = self.chunks[int(index_id)]
            if active_filters and not self._matches_filters(chunk, active_filters):
                continue
            results.append(
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(score),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def keyword_search(
        self,
        query: str,
        top_k: int,
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise VectorStoreError("top_k must be greater than zero.")

        query_terms = _tokenize(query)
        if not query_terms:
            return []

        active_filters = _clean_filters(filters)
        candidates = [
            chunk for chunk in self.chunks if not active_filters or self._matches_filters(chunk, active_filters)
        ]
        if not candidates:
            return []

        tokenized: list[list[str]] = [
            _tokenize(f"{chunk.get('text', '')} {_metadata_search_text(chunk.get('metadata', {}), self.documents)}")
            for chunk in candidates
        ]
        lengths = [len(tokens) or 1 for tokens in tokenized]
        avg_length = sum(lengths) / max(1, len(lengths))

        document_frequency: Counter[str] = Counter()
        for tokens in tokenized:
            document_frequency.update(set(tokens))

        query_counts = Counter(query_terms)
        raw_scores: list[tuple[float, dict[str, Any]]] = []
        k1 = 1.5
        b = 0.75
        total_docs = len(candidates)

        for chunk, tokens, doc_length in zip(candidates, tokenized, lengths):
            term_counts = Counter(tokens)
            score = 0.0
            for term, query_count in query_counts.items():
                frequency = term_counts.get(term, 0)
                if frequency <= 0:
                    continue
                idf = math.log(1 + (total_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
                denominator = frequency + k1 * (1 - b + b * (doc_length / avg_length))
                score += query_count * idf * ((frequency * (k1 + 1)) / denominator)

            if score > 0:
                raw_scores.append((score, chunk))

        if not raw_scores:
            return []

        max_score = max(score for score, _ in raw_scores) or 1.0
        ranked = sorted(raw_scores, key=lambda item: item[0], reverse=True)[:top_k]
        results: list[dict[str, Any]] = []
        for score, chunk in ranked:
            results.append(
                {
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": round(score / max_score, 6),
                    "raw_keyword_score": score,
                }
            )
        return results

    def reset(self) -> None:
        self.index = None
        self.chunks = []
        self.documents = {}
        self.save()
        logger.info("Reset FAISS index at %s", self.settings.index_dir)

    def remove_document(self, file_hash: str) -> bool:
        if file_hash not in self.documents:
            return False

        if self.index is None:
            self.chunks = [
                chunk for chunk in self.chunks if chunk.get("metadata", {}).get("file_hash") != file_hash
            ]
            self.documents.pop(file_hash, None)
            self.save()
            return True

        keep_positions = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.get("metadata", {}).get("file_hash") != file_hash
        ]

        kept_chunks: list[dict[str, Any]] = []
        kept_vectors: list[np.ndarray] = []
        for old_position in keep_positions:
            chunk = self.chunks[old_position]
            metadata = dict(chunk["metadata"])
            metadata["vector_position"] = len(kept_chunks)
            kept_chunks.append({**chunk, "metadata": metadata})
            kept_vectors.append(self.index.reconstruct(old_position))

        self.documents.pop(file_hash, None)
        if kept_vectors:
            vectors = np.asarray(kept_vectors, dtype="float32")
            new_index = faiss.IndexFlatIP(vectors.shape[1])
            new_index.add(vectors)
            self.index = new_index
        else:
            self.index = None

        self.chunks = kept_chunks
        self.documents = self._rebuild_document_ranges()
        self.save()
        logger.info("Removed document %s from %s", file_hash, self.settings.index_dir)
        return True

    def _matches_filters(self, chunk: dict[str, Any], filters: dict[str, Any]) -> bool:
        metadata = chunk.get("metadata", {})
        document = self.documents.get(metadata.get("file_hash"), {})

        document_hashes = set(filters.get("document_hashes") or [])
        if document_hashes and metadata.get("file_hash") not in document_hashes:
            return False

        file_types = {item.lower().lstrip(".") for item in filters.get("file_types") or []}
        if file_types:
            file_type = Path(metadata.get("file_name") or document.get("file_name") or "").suffix.lower().lstrip(".")
            if file_type not in file_types:
                return False

        source_types = {item.lower() for item in filters.get("source_types") or []}
        if source_types and str(metadata.get("source_type") or "text").lower() not in source_types:
            return False

        uploaded_at = document.get("uploaded_at")
        if filters.get("uploaded_after") and not _date_is_after_or_equal(uploaded_at, filters["uploaded_after"]):
            return False
        if filters.get("uploaded_before") and not _date_is_before_or_equal(uploaded_at, filters["uploaded_before"]):
            return False

        path_query = str(filters.get("path_query") or "").strip().lower()
        if path_query and path_query not in str(document.get("source_path") or metadata.get("source_path") or "").lower():
            return False

        metadata_query = str(filters.get("metadata_query") or "").strip().lower()
        if metadata_query and metadata_query not in _metadata_search_text(metadata, self.documents).lower():
            return False

        return True

    def _ensure_index(self, dimension: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)
            return

        if self.index.d != dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch. Existing index dimension={self.index.d}, "
                f"new embedding dimension={dimension}."
            )

    def _rebuild_document_ranges(self) -> dict[str, dict[str, Any]]:
        rebuilt: dict[str, dict[str, Any]] = {}

        for position, chunk in enumerate(self.chunks):
            metadata = chunk.get("metadata", {})
            file_hash = metadata.get("file_hash")
            if not file_hash:
                continue

            existing = self.documents.get(file_hash, {})
            if file_hash not in rebuilt:
                rebuilt[file_hash] = {
                    **existing,
                    "file_hash": file_hash,
                    "file_name": metadata.get("file_name", existing.get("file_name")),
                    "chunk_count": 0,
                    "visual_chunk_count": 0,
                    "vector_start": position,
                    "vector_end": position,
                    "embedding_model": metadata.get(
                        "embedding_model",
                        existing.get("embedding_model", self.settings.openai_embedding_model),
                    ),
                    "source_path": metadata.get("source_path", existing.get("source_path")),
                    "uploaded_at": existing.get("uploaded_at", datetime.now(timezone.utc).isoformat()),
                }

            rebuilt[file_hash]["chunk_count"] += 1
            if metadata.get("source_type") == "image":
                rebuilt[file_hash]["visual_chunk_count"] += 1
            rebuilt[file_hash]["vector_end"] = position

        return rebuilt

    @staticmethod
    def _normalize_vectors(embeddings: list[list[float]]) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[0] == 0:
            raise VectorStoreError("Embeddings must be a non-empty 2D array.")

        faiss.normalize_L2(vectors)
        return vectors

    @staticmethod
    def _read_json_list(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise VectorStoreError(f"Expected a JSON list in {path}.")
        return data

    @staticmethod
    def _read_json_dict(path: Path) -> dict[str, dict[str, Any]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise VectorStoreError(f"Expected a JSON object in {path}.")
        return data

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        temp_path.replace(path)


def _clean_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    if not filters:
        return {}
    return {key: value for key, value in filters.items() if value not in (None, "", [], {})}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w.-]+", text.lower(), flags=re.UNICODE)


def _metadata_search_text(metadata: dict[str, Any], documents: dict[str, dict[str, Any]]) -> str:
    document = documents.get(metadata.get("file_hash"), {})
    values = [
        metadata.get("file_name"),
        metadata.get("file_hash"),
        metadata.get("source_type"),
        metadata.get("source_path"),
        document.get("file_name"),
        document.get("source_path"),
        document.get("uploaded_at"),
        " ".join(str(tag) for tag in document.get("tags", []) if tag),
    ]
    return " ".join(str(value) for value in values if value)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _date_is_after_or_equal(uploaded_at: str | None, boundary: str) -> bool:
    uploaded = _parse_datetime(uploaded_at)
    lower = _parse_datetime(boundary)
    if not uploaded or not lower:
        return True
    return uploaded >= lower


def _date_is_before_or_equal(uploaded_at: str | None, boundary: str) -> bool:
    uploaded = _parse_datetime(uploaded_at)
    upper = _parse_datetime(boundary)
    if not uploaded or not upper:
        return True
    return uploaded <= upper
