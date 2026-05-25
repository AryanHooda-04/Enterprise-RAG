from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings, settings_for_models
from errors import RAGApplicationError
from ingestion import ingest_file, safe_filename
from rag_pipeline import RAGPipeline
from vector_store import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Local RAG API",
    version="1.0.0",
    description=(
        "Upload PDF, TXT, DOCX, and image files, index them with FAISS, "
        "and ask context-grounded questions."
    ),
)

vector_store = VectorStore(settings)
rag_pipeline = RAGPipeline(vector_store, settings)
store_lock = Lock()


class UploadResponse(BaseModel):
    file_name: str
    file_hash: str
    chunks_added: int
    total_chunks: int
    skipped: bool


class UploadBatchResponse(BaseModel):
    message: str
    files_processed: int
    chunks_added: int
    duplicates_skipped: int
    results: list[UploadResponse]


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    model: str | None = None
    embedding_model: str | None = None
    response_language: str | None = None
    search_mode: str | None = Field(default="hybrid")
    filters: dict | None = None


class SourceMetadata(BaseModel):
    file_name: str | None
    file_hash: str | None = None
    chunk_id: str | None = None
    page_number: int | None
    source_type: str | None = None
    image_index: int | None = None
    chunk_index: int | None
    token_start: int | None = None
    token_count: int | None = None
    vector_position: int | None = None
    embedding_model: str | None = None
    source_path: str | None = None
    retrieval_method: str | None = None
    semantic_score: float | None = None
    keyword_score: float | None = None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
    source_metadata: list[SourceMetadata]


@app.exception_handler(RAGApplicationError)
async def rag_error_handler(_, exc: RAGApplicationError):
    logger.warning("RAG error: %s", exc.message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(Exception)
async def unexpected_error_handler(_, exc: Exception):
    logger.exception("Unexpected failure: %s", exc)
    return JSONResponse(status_code=500, content={"detail": "Unexpected server error."})


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "indexed_chunks": vector_store.total_vectors,
        "documents": len(vector_store.documents),
    }


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": vector_store.list_documents()}


def _format_size(megabytes: int) -> str:
    if megabytes >= 1024:
        return f"{megabytes / 1024:g} GB"
    return f"{megabytes} MB"


def _role_from_header(role: str | None) -> str:
    return (role or settings.default_role or "User").strip().lower()


def _is_admin(role: str | None, admin_key: str | None = None) -> bool:
    if settings.auth_enabled and settings.admin_password:
        return admin_key == settings.admin_password
    return _role_from_header(role) == "admin"


def require_admin(
    x_rag_role: str | None = Header(default=None),
    x_rag_admin_key: str | None = Header(default=None),
) -> None:
    if not _is_admin(x_rag_role, x_rag_admin_key):
        raise HTTPException(status_code=403, detail="Admin role required.")


async def _persist_upload_file(file: UploadFile) -> tuple[Path, str, str]:
    original_name = safe_filename(file.filename)
    extension = Path(original_name).suffix.lower()

    if extension not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Allowed: {', '.join(settings.allowed_extensions)}.",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    settings.ensure_directories()
    temp_path = settings.upload_dir / f"pending_{uuid4().hex}_{original_name}"
    digest = hashlib.sha256()
    size = 0

    with temp_path.open("wb") as output:
        while True:
            block = await file.read(1024 * 1024)
            if not block:
                break

            size += len(block)
            if size > max_bytes:
                output.close()
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=f"{original_name} exceeds the {_format_size(settings.max_upload_size_mb)} per-file limit.",
                )

            digest.update(block)
            output.write(block)

    if size == 0:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"{original_name} is empty.")

    file_hash = digest.hexdigest()
    saved_path = settings.upload_dir / f"{file_hash[:12]}_{original_name}"
    temp_path.replace(saved_path)
    return saved_path, file_hash, original_name


def _duplicate_upload_response(store: VectorStore, file_hash: str, original_name: str) -> UploadResponse:
    document = store.get_document(file_hash) or {}
    logger.info("Duplicate upload skipped for %s", original_name)
    return UploadResponse(
        file_name=document.get("file_name", original_name),
        file_hash=file_hash,
        chunks_added=0,
        total_chunks=document.get("chunk_count", 0),
        skipped=True,
    )


def _ingest_saved_file(
    saved_path: Path,
    *,
    store: VectorStore,
    runtime_settings,
    file_hash: str,
    original_name: str,
    chunk_size_tokens: int | None,
    chunk_overlap_tokens: int | None,
) -> UploadResponse:
    if store.has_document(file_hash):
        saved_path.unlink(missing_ok=True)
        return _duplicate_upload_response(store, file_hash, original_name)

    with store_lock:
        result = ingest_file(
            saved_path,
            store,
            file_hash=file_hash,
            display_name=original_name,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            active_settings=runtime_settings,
        )

    return UploadResponse(**result)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size_tokens: int | None = Form(default=None),
    chunk_overlap_tokens: int | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    vision_model: str | None = Form(default=None),
    vision_enabled: bool | None = Form(default=None),
    vision_detail: str | None = Form(default=None),
    _: None = Depends(require_admin),
) -> UploadResponse:
    runtime_settings = settings_for_models(
        embedding_model=embedding_model,
        vision_model=vision_model,
        vision_ingestion_enabled=vision_enabled,
        vision_detail=vision_detail,
    )
    store = VectorStore(runtime_settings)
    saved_path, file_hash, original_name = await _persist_upload_file(file)
    return _ingest_saved_file(
        saved_path,
        store=store,
        runtime_settings=runtime_settings,
        file_hash=file_hash,
        original_name=original_name,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
    )


@app.post("/upload/batch", response_model=UploadBatchResponse)
async def upload_documents(
    files: list[UploadFile] = File(...),
    chunk_size_tokens: int | None = Form(default=None),
    chunk_overlap_tokens: int | None = Form(default=None),
    embedding_model: str | None = Form(default=None),
    vision_model: str | None = Form(default=None),
    vision_enabled: bool | None = Form(default=None),
    vision_detail: str | None = Form(default=None),
    _: None = Depends(require_admin),
) -> UploadBatchResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded.")

    if len(files) > settings.max_upload_files:
        raise HTTPException(
            status_code=400,
            detail=f"Upload {settings.max_upload_files} documents or fewer at a time.",
        )

    results: list[UploadResponse] = []
    runtime_settings = settings_for_models(
        embedding_model=embedding_model,
        vision_model=vision_model,
        vision_ingestion_enabled=vision_enabled,
        vision_detail=vision_detail,
    )
    store = VectorStore(runtime_settings)
    for file in files:
        saved_path, file_hash, original_name = await _persist_upload_file(file)
        results.append(
            _ingest_saved_file(
                saved_path,
                store=store,
                runtime_settings=runtime_settings,
                file_hash=file_hash,
                original_name=original_name,
                chunk_size_tokens=chunk_size_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
            )
        )

    chunks_added = sum(result.chunks_added for result in results)
    duplicates_skipped = sum(1 for result in results if result.skipped)
    return UploadBatchResponse(
        message="Indexing completed.",
        files_processed=len(results),
        chunks_added=chunks_added,
        duplicates_skipped=duplicates_skipped,
        results=results,
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(
    request: AskRequest,
    x_rag_role: str | None = Header(default=None),
    x_rag_admin_key: str | None = Header(default=None),
) -> AskResponse:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if (request.model or request.embedding_model) and not _is_admin(x_rag_role, x_rag_admin_key):
        raise HTTPException(status_code=403, detail="Admin role required to override models.")

    logger.info("Received query request")
    runtime_settings = settings_for_models(
        chat_model=request.model,
        embedding_model=request.embedding_model,
    )
    store = VectorStore(runtime_settings)
    pipeline = RAGPipeline(store, runtime_settings)
    result = pipeline.answer_question(
        query,
        top_k=request.top_k,
        min_score=request.min_score,
        response_language=request.response_language,
        filters=request.filters,
        search_mode=request.search_mode or "hybrid",
    )
    return AskResponse(**result)
