from __future__ import annotations

import logging
import re
from pathlib import Path

from config import Settings, settings
from errors import RetrievalError
from ingestion import generate_embeddings
from vector_store import VectorStore


logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, active_settings: Settings = settings):
        self.vector_store = vector_store
        self.settings = active_settings

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        filters: dict | None = None,
        search_mode: str = "hybrid",
    ) -> list[dict]:
        clean_query = query.strip()
        if not clean_query:
            raise RetrievalError("Query cannot be empty.")

        if self.vector_store.is_empty:
            raise RetrievalError("No documents have been indexed yet. Upload a document first.")

        resolved_top_k = top_k or self.settings.top_k
        if resolved_top_k <= 0:
            raise RetrievalError("top_k must be greater than zero.")

        mode = (search_mode or "hybrid").strip().lower()
        if mode not in {"semantic", "keyword", "hybrid"}:
            raise RetrievalError("search_mode must be one of: semantic, keyword, hybrid.")

        active_filters = dict(filters or {})
        if not active_filters.get("document_hashes"):
            inferred_document_hashes = _infer_document_hashes(clean_query, self.vector_store.documents)
            if inferred_document_hashes:
                active_filters["document_hashes"] = inferred_document_hashes

        logger.info("Retrieving top %s chunks for query using %s search", resolved_top_k, mode)
        expanded_top_k = max(resolved_top_k, resolved_top_k * 3)
        semantic_results: list[dict] = []
        keyword_results: list[dict] = []

        if mode in {"semantic", "hybrid"}:
            query_embedding = generate_embeddings([clean_query], active_settings=self.settings)[0]
            semantic_results = self.vector_store.search(
                query_embedding,
                expanded_top_k,
                filters=active_filters,
            )

        if mode in {"keyword", "hybrid"}:
            keyword_results = self.vector_store.keyword_search(
                clean_query,
                expanded_top_k,
                filters=active_filters,
            )

        if mode == "semantic":
            return _annotate_semantic(semantic_results)[:resolved_top_k]
        if mode == "keyword":
            return _annotate_keyword(keyword_results)[:resolved_top_k]
        return _merge_hybrid_results(
            semantic_results,
            keyword_results,
            top_k=resolved_top_k,
            semantic_weight=self.settings.hybrid_semantic_weight,
        )


def semantic_search(
    query: str,
    vector_store: VectorStore,
    *,
    top_k: int | None = None,
    filters: dict | None = None,
    active_settings: Settings = settings,
) -> list[dict]:
    return Retriever(vector_store, active_settings).retrieve(
        query,
        top_k=top_k,
        filters=filters,
        search_mode="semantic",
    )


def _annotate_semantic(results: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for result in results:
        score = _clamp_score(result.get("score", 0.0))
        annotated.append(
            {
                **result,
                "score": score,
                "semantic_score": score,
                "keyword_score": 0.0,
                "retrieval_method": "semantic",
            }
        )
    return annotated


def _annotate_keyword(results: list[dict]) -> list[dict]:
    annotated: list[dict] = []
    for result in results:
        score = _clamp_score(result.get("score", 0.0))
        annotated.append(
            {
                **result,
                "score": score,
                "semantic_score": 0.0,
                "keyword_score": score,
                "retrieval_method": "keyword",
            }
        )
    return annotated


def _merge_hybrid_results(
    semantic_results: list[dict],
    keyword_results: list[dict],
    *,
    top_k: int,
    semantic_weight: float,
) -> list[dict]:
    semantic_weight = max(0.0, min(1.0, semantic_weight))
    keyword_weight = 1.0 - semantic_weight
    merged: dict[str, dict] = {}

    for result in _annotate_semantic(semantic_results):
        key = _result_key(result)
        merged[key] = {
            **result,
            "keyword_score": 0.0,
        }

    for result in _annotate_keyword(keyword_results):
        key = _result_key(result)
        if key in merged:
            merged[key]["keyword_score"] = result["keyword_score"]
            merged[key]["raw_keyword_score"] = result.get("raw_keyword_score")
        else:
            merged[key] = result

    ranked: list[dict] = []
    for result in merged.values():
        semantic_score = _clamp_score(result.get("semantic_score", 0.0))
        keyword_score = _clamp_score(result.get("keyword_score", 0.0))
        combined = round((semantic_weight * semantic_score) + (keyword_weight * keyword_score), 6)
        method = "hybrid" if semantic_score and keyword_score else ("semantic" if semantic_score else "keyword")
        ranked.append(
            {
                **result,
                "score": combined,
                "semantic_score": round(semantic_score, 6),
                "keyword_score": round(keyword_score, 6),
                "retrieval_method": method,
            }
        )

    return sorted(ranked, key=lambda item: item.get("score", 0.0), reverse=True)[:top_k]


def _result_key(result: dict) -> str:
    metadata = result.get("metadata", {})
    return str(
        metadata.get("chunk_id")
        or metadata.get("vector_position")
        or f"{metadata.get('file_hash')}:{metadata.get('chunk_index')}"
    )


def _clamp_score(value: float) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


DOCUMENT_TITLE_STOPWORDS = {
    "a",
    "an",
    "and",
    "book",
    "by",
    "ebook",
    "edition",
    "pdf",
    "story",
    "the",
}


def _infer_document_hashes(query: str, documents: dict[str, dict]) -> list[str]:
    query_terms = set(_document_title_tokens(query))
    if not query_terms:
        return []

    candidates: list[tuple[int, str]] = []
    for file_hash, document in documents.items():
        title_terms = set(_document_title_tokens(Path(document.get("file_name", "")).stem))
        if not title_terms:
            continue

        overlap = len(query_terms & title_terms)
        if overlap >= 2 or (len(title_terms) == 1 and overlap == 1):
            candidates.append((overlap, file_hash))

    if not candidates:
        return []

    best_score = max(score for score, _ in candidates)
    return [file_hash for score, file_hash in candidates if score == best_score]


def _document_title_tokens(value: str) -> list[str]:
    tokens = _tokenize(value.replace("_", " ").replace("-", " "))
    return [
        token
        for token in tokens
        if token not in DOCUMENT_TITLE_STOPWORDS and len(token) > 1 and not token.isdigit()
    ]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w.-]+", text.lower(), flags=re.UNICODE)
