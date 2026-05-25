from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import tiktoken
from openai import BadRequestError

from config import Settings, settings
from errors import LLMError, clean_external_error
from openai_client import get_openai_client
from retriever import Retriever
from usage_store import record_openai_usage, record_usage
from vector_store import VectorStore


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """You are a helpful assistant.
Answer ONLY from the provided context.
If answer is not in context, say "I don't know" in the required answer language.
Always answer in the required answer language.
For summary or "what is this about" questions, synthesize only from the provided context.
If the context supports more than one interpretation, answer with the supported options instead of guessing.

Required answer language:
{language_instruction}

Context:
{context}

Question:
{question}
"""


class RAGPipeline:
    def __init__(self, vector_store: VectorStore, active_settings: Settings = settings):
        self.settings = active_settings
        self.vector_store = vector_store
        self.retriever = Retriever(vector_store, active_settings)

    def answer_question(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        response_language: str | None = None,
        filters: dict[str, Any] | None = None,
        search_mode: str = "hybrid",
    ) -> dict[str, Any]:
        logger.info("Handling query request")
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            search_mode=search_mode,
        )
        if min_score is not None:
            retrieved_chunks = [
                chunk for chunk in retrieved_chunks if float(chunk.get("score", 0.0)) >= min_score
            ]
        retrieved_chunks = self._expand_context_window(
            retrieved_chunks,
            top_k or self.settings.top_k,
            question=question,
        )

        if not retrieved_chunks:
            return {
                "answer": _unknown_answer(response_language, question),
                "sources": [],
                "source_metadata": [],
                "confidence": 0.0,
            }

        context = _format_context(retrieved_chunks)
        prompt = PROMPT_TEMPLATE.format(
            context=context,
            question=question.strip(),
            language_instruction=_language_instruction(response_language, question),
        )
        answer = self._generate_answer(prompt)

        sources = [chunk["text"] for chunk in retrieved_chunks]
        source_metadata = [_source_metadata(chunk) for chunk in retrieved_chunks]
        confidence = _confidence_from_results(retrieved_chunks)

        return {
            "answer": answer,
            "sources": sources,
            "source_metadata": source_metadata,
            "confidence": confidence,
        }

    def retrieve_chunks(
        self,
        question: str,
        top_k: int | None = None,
        min_score: float | None = None,
        filters: dict[str, Any] | None = None,
        search_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            search_mode=search_mode,
        )
        if min_score is not None:
            retrieved_chunks = [
                chunk for chunk in retrieved_chunks if float(chunk.get("score", 0.0)) >= min_score
            ]
        return self._expand_context_window(
            retrieved_chunks,
            top_k or self.settings.top_k,
            question=question,
        )

    def build_prompt(self, question: str, chunks: list[dict[str, Any]], response_language: str | None = None) -> str:
        context = _format_context(chunks)
        return PROMPT_TEMPLATE.format(
            context=context,
            question=question.strip(),
            language_instruction=_language_instruction(response_language, question),
        )

    def result_from_answer(self, answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "answer": answer.strip() or "I don't know",
            "sources": [chunk["text"] for chunk in chunks],
            "source_metadata": [_source_metadata(chunk) for chunk in chunks],
            "confidence": _confidence_from_results(chunks),
        }

    def _expand_context_window(
        self,
        chunks: list[dict[str, Any]],
        top_k: int,
        *,
        question: str = "",
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        overview_mode = _is_overview_query(question)
        max_chunks = max(top_k, min(top_k + (8 if overview_mode else 4), top_k * 3))
        lookup: dict[tuple[str, int], dict[str, Any]] = {}
        for record in self.vector_store.chunks:
            metadata = record.get("metadata", {})
            file_hash = metadata.get("file_hash")
            chunk_index = metadata.get("chunk_index")
            if file_hash is None or chunk_index is None:
                continue
            try:
                lookup[(str(file_hash), int(chunk_index))] = record
            except (TypeError, ValueError):
                continue

        expanded: list[dict[str, Any]] = []
        seen: set[str] = set()

        def result_key(result: dict[str, Any]) -> str:
            metadata = result.get("metadata", {})
            return str(
                metadata.get("chunk_id")
                or metadata.get("vector_position")
                or f"{metadata.get('file_hash')}:{metadata.get('chunk_index')}"
            )

        def add_result(result: dict[str, Any]) -> None:
            if len(expanded) >= max_chunks:
                return
            key = result_key(result)
            if key in seen:
                return
            seen.add(key)
            expanded.append(result)

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            file_hash = metadata.get("file_hash")
            chunk_index = metadata.get("chunk_index")
            if file_hash is None or chunk_index is None:
                add_result(chunk)
                continue

            try:
                center = int(chunk_index)
            except (TypeError, ValueError):
                add_result(chunk)
                continue

            for offset in (-1, 0, 1):
                if len(expanded) >= max_chunks:
                    break
                if offset == 0:
                    add_result(chunk)
                    continue

                neighbor = lookup.get((str(file_hash), center + offset))
                if not neighbor:
                    continue

                neighbor_score = max(0.0, min(1.0, float(chunk.get("score", 0.0)) * 0.97))
                add_result(
                    {
                        "text": neighbor.get("text", ""),
                        "metadata": neighbor.get("metadata", {}),
                        "score": neighbor_score,
                        "semantic_score": chunk.get("semantic_score", 0.0),
                        "keyword_score": chunk.get("keyword_score", 0.0),
                        "retrieval_method": "context-window",
                    }
                )

        if overview_mode:
            expanded = self._prepend_document_overview(expanded, max_chunks)

        return expanded

    def _prepend_document_overview(self, chunks: list[dict[str, Any]], max_chunks: int) -> list[dict[str, Any]]:
        file_hashes = [
            chunk.get("metadata", {}).get("file_hash")
            for chunk in chunks
            if chunk.get("metadata", {}).get("file_hash")
        ]
        if not file_hashes:
            return chunks

        dominant_hash = Counter(file_hashes).most_common(1)[0][0]
        overview_records = sorted(
            [
                record
                for record in self.vector_store.chunks
                if record.get("metadata", {}).get("file_hash") == dominant_hash
            ],
            key=lambda record: int(record.get("metadata", {}).get("chunk_index") or 0),
        )[:5]

        best_score = max(float(chunk.get("score", 0.0)) for chunk in chunks)
        overview_chunks = [
            {
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
                "score": max(0.0, min(1.0, best_score * 0.96)),
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "retrieval_method": "document-overview",
            }
            for record in overview_records
        ]

        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk in overview_chunks + chunks:
            metadata = chunk.get("metadata", {})
            key = str(
                metadata.get("chunk_id")
                or metadata.get("vector_position")
                or f"{metadata.get('file_hash')}:{metadata.get('chunk_index')}"
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(chunk)
            if len(merged) >= max_chunks:
                break
        return merged

    def stream_answer(self, prompt: str):
        client = get_openai_client()
        params: dict[str, Any] = {
            "model": self.settings.openai_chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.settings.max_answer_tokens,
            "stream": True,
        }
        if self.settings.openai_temperature is not None:
            params["temperature"] = self.settings.openai_temperature

        prompt_token_estimate = _estimate_tokens(prompt, self.settings.openai_chat_model)
        completion_parts: list[str] = []
        try:
            stream = self._create_chat_completion_with_fallbacks(client, params)
            for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None)
                if content:
                    completion_parts.append(content)
                    yield content
        except Exception as exc:
            raise LLMError(clean_external_error(exc, "Streaming answer generation")) from exc
        finally:
            if completion_parts:
                completion = "".join(completion_parts)
                record_usage(
                    "chat_stream",
                    self.settings.openai_chat_model,
                    active_settings=self.settings,
                    prompt_tokens=prompt_token_estimate,
                    completion_tokens=_estimate_tokens(completion, self.settings.openai_chat_model),
                    metadata={"streamed": True},
                )

    def _generate_answer(self, prompt: str) -> str:
        client = get_openai_client()
        params: dict[str, Any] = {
            "model": self.settings.openai_chat_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.settings.max_answer_tokens,
        }
        if self.settings.openai_temperature is not None:
            params["temperature"] = self.settings.openai_temperature

        try:
            response = self._create_chat_completion_with_fallbacks(client, params)
            record_openai_usage(
                "chat",
                self.settings.openai_chat_model,
                getattr(response, "usage", None),
                active_settings=self.settings,
            )
            answer = response.choices[0].message.content or ""
            return answer.strip() or "I don't know"
        except Exception as exc:
            raise LLMError(clean_external_error(exc, "Answer generation")) from exc

    @staticmethod
    def _create_chat_completion_with_fallbacks(client, params: dict[str, Any]):
        last_error: Exception | None = None

        for _ in range(3):
            try:
                return client.chat.completions.create(**params)
            except BadRequestError as exc:
                last_error = exc
                message = str(exc).lower()
                changed = False

                if "max_completion_tokens" in message and "max_tokens" not in params:
                    params["max_tokens"] = params.pop("max_completion_tokens")
                    changed = True

                if "temperature" in message and "temperature" in params:
                    params.pop("temperature")
                    changed = True

                if not changed:
                    raise

        if last_error:
            raise last_error
        raise LLMError("Answer generation failed.")


def _format_context(chunks: list[dict[str, Any]]) -> str:
    formatted: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata", {})
        file_name = metadata.get("file_name", "unknown")
        page_number = metadata.get("page_number")
        page_label = f", page {page_number}" if page_number else ""
        formatted.append(f"[Source {index}: {file_name}{page_label}]\n{chunk['text']}")
    return "\n\n".join(formatted)


def _source_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata", {})
    return {
        "file_name": metadata.get("file_name"),
        "file_hash": metadata.get("file_hash"),
        "chunk_id": metadata.get("chunk_id"),
        "page_number": metadata.get("page_number"),
        "source_type": metadata.get("source_type"),
        "image_index": metadata.get("image_index"),
        "chunk_index": metadata.get("chunk_index"),
        "token_start": metadata.get("token_start"),
        "token_count": metadata.get("token_count"),
        "vector_position": metadata.get("vector_position"),
        "embedding_model": metadata.get("embedding_model"),
        "source_path": metadata.get("source_path"),
        "retrieval_method": chunk.get("retrieval_method"),
        "semantic_score": round(float(chunk.get("semantic_score", 0.0)), 4),
        "keyword_score": round(float(chunk.get("keyword_score", 0.0)), 4),
        "score": round(float(chunk.get("score", 0.0)), 4),
    }


def _confidence_from_results(chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0

    best_score = max(float(chunk.get("score", 0.0)) for chunk in chunks)
    return round(max(0.0, min(1.0, best_score)), 4)


def _is_overview_query(question: str) -> bool:
    normalized = f" {question.strip().lower()} "
    return any(
        phrase in normalized
        for phrase in (
            " all about",
            " story",
            " summary",
            " summarize",
            " talks about",
            " talk about",
            " theme",
            " overview",
            " plot",
        )
    )


def _language_instruction(response_language: str | None, question: str) -> str:
    if response_language and response_language != "Auto":
        return response_language
    inferred = _infer_language(question)
    if inferred != "English":
        return inferred
    return "the same language as the user's current question"


def _unknown_answer(response_language: str | None, question: str) -> str:
    language = response_language if response_language and response_language != "Auto" else _infer_language(question)
    return {
        "Arabic": "\u0644\u0627 \u0623\u0639\u0631\u0641",
        "Chinese": "\u6211\u4e0d\u77e5\u9053",
        "French": "Je ne sais pas",
        "German": "Ich wei\u00df es nicht",
        "Hindi": "\u092e\u0941\u091d\u0947 \u0928\u0939\u0940\u0902 \u092a\u0924\u093e",
        "Japanese": "\u308f\u304b\u308a\u307e\u305b\u3093",
        "Korean": "\ubaa8\ub974\uaca0\uc2b5\ub2c8\ub2e4",
        "Portuguese": "N\u00e3o sei",
        "Spanish": "No lo s\u00e9",
    }.get(language, "I don't know")


def _infer_language(text: str) -> str:
    if any("\u0900" <= char <= "\u097f" for char in text):
        return "Hindi"
    if any("\u0600" <= char <= "\u06ff" for char in text):
        return "Arabic"
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return "Chinese"
    if any("\u3040" <= char <= "\u30ff" for char in text):
        return "Japanese"
    if any("\uac00" <= char <= "\ud7af" for char in text):
        return "Korean"
    return "English"


def _estimate_tokens(text: str, model_name: str) -> int:
    try:
        tokenizer = tiktoken.encoding_for_model(model_name)
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text or ""))
