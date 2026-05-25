from __future__ import annotations

import os
import re
import ssl
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import certifi


BASE_DIR = Path(__file__).resolve().parent


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")

        if key:
            os.environ.setdefault(key, value)


_load_local_env(BASE_DIR / ".env")

DEFAULT_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
DEFAULT_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
DEFAULT_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4.1-mini")
DEFAULT_TRANSCRIPTION_MODEL = os.getenv("OPENAI_TRANSCRIPTION_MODEL", "gpt-4o-mini-transcribe")
DEFAULT_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
DEFAULT_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
DEFAULT_DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data")).resolve()

CHAT_MODEL_OPTIONS = (
    "gpt-5.2",
    "gpt-5.2-chat-latest",
    "gpt-5.1",
    "gpt-5.1-chat-latest",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
)

EMBEDDING_MODEL_OPTIONS = (
    "text-embedding-3-large",
    "text-embedding-3-small",
    "text-embedding-ada-002",
)

VISION_MODEL_OPTIONS = (
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5.2",
    "gpt-5",
)

AUDIO_TRANSCRIPTION_MODEL_OPTIONS = (
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "whisper-1",
)

TTS_MODEL_OPTIONS = (
    "gpt-4o-mini-tts",
    "tts-1",
    "tts-1-hd",
)

TTS_VOICE_OPTIONS = (
    "marin",
    "cedar",
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)

VOICE_LANGUAGE_OPTIONS = (
    "Auto",
    "English",
    "Hindi",
    "Spanish",
    "French",
    "German",
    "Arabic",
    "Chinese",
    "Japanese",
    "Korean",
    "Portuguese",
)


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "default"


DEFAULT_INDEX_DIR = Path(
    os.getenv("RAG_INDEX_DIR", DEFAULT_DATA_DIR / "index" / _safe_path_part(DEFAULT_EMBEDDING_MODEL))
).resolve()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _env_optional_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, none, or null") from exc


def _resolve_ca_bundle() -> str:
    return (
        os.getenv("OPENAI_CA_BUNDLE")
        or os.getenv("SSL_CERT_FILE")
        or os.getenv("REQUESTS_CA_BUNDLE")
        or certifi.where()
    )


def _resolve_ssl_mode() -> str:
    explicit_mode = os.getenv("OPENAI_SSL_MODE")
    if explicit_mode:
        return explicit_mode.strip().lower()
    if not _env_bool("OPENAI_VERIFY_SSL", True):
        return "insecure"
    if os.getenv("OPENAI_CA_BUNDLE") or os.getenv("SSL_CERT_FILE") or os.getenv("REQUESTS_CA_BUNDLE"):
        return "custom"
    return "system"


@dataclass(frozen=True)
class Settings:
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    openai_chat_model: str = DEFAULT_CHAT_MODEL
    openai_vision_model: str = DEFAULT_VISION_MODEL
    openai_transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    openai_tts_model: str = DEFAULT_TTS_MODEL
    openai_tts_voice: str = DEFAULT_TTS_VOICE
    openai_timeout_seconds: float = _env_float("OPENAI_TIMEOUT_SECONDS", 60.0)
    openai_max_retries: int = _env_int("OPENAI_MAX_RETRIES", 2)
    openai_temperature: float | None = _env_optional_float("OPENAI_TEMPERATURE", 0.0)
    openai_ssl_mode: str = _resolve_ssl_mode()
    openai_ca_bundle: str = _resolve_ca_bundle()
    openai_verify_ssl: bool = _env_bool("OPENAI_VERIFY_SSL", True)
    disable_ssl_warnings: bool = _env_bool("DISABLE_SSL_WARNINGS", False)
    empty_ca_bundle_fallback: bool = _env_bool("RAG_EMPTY_CA_BUNDLE_FALLBACK", False)

    chunk_size_tokens: int = _env_int("CHUNK_SIZE_TOKENS", 500)
    chunk_overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", 50)
    top_k: int = _env_int("TOP_K", 5)
    max_answer_tokens: int = _env_int("MAX_ANSWER_TOKENS", 800)
    embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 64)
    hybrid_semantic_weight: float = _env_float("HYBRID_SEMANTIC_WEIGHT", 0.7)
    vision_ingestion_enabled: bool = _env_bool("VISION_INGESTION_ENABLED", True)
    vision_detail: str = os.getenv("VISION_DETAIL", "high")
    max_visual_pages: int = _env_int("MAX_VISUAL_PAGES", 80)
    max_docx_images: int = _env_int("MAX_DOCX_IMAGES", 80)
    max_image_dimension_px: int = _env_int("MAX_IMAGE_DIMENSION_PX", 1600)
    voice_output_enabled: bool = _env_bool("VOICE_OUTPUT_ENABLED", True)
    default_voice_language: str = os.getenv("VOICE_LANGUAGE", "Auto")

    data_dir: Path = DEFAULT_DATA_DIR
    upload_dir: Path = Path(os.getenv("RAG_UPLOAD_DIR", DEFAULT_DATA_DIR / "uploads")).resolve()
    index_dir: Path = DEFAULT_INDEX_DIR
    index_path: Path = Path(os.getenv("RAG_FAISS_INDEX_PATH", DEFAULT_INDEX_DIR / "index.faiss")).resolve()
    chunks_path: Path = Path(os.getenv("RAG_CHUNKS_PATH", DEFAULT_INDEX_DIR / "chunks.json")).resolve()
    documents_path: Path = Path(os.getenv("RAG_DOCUMENTS_PATH", DEFAULT_INDEX_DIR / "documents.json")).resolve()
    feedback_path: Path = Path(os.getenv("RAG_FEEDBACK_PATH", DEFAULT_DATA_DIR / "feedback.jsonl")).resolve()
    usage_path: Path = Path(os.getenv("RAG_USAGE_PATH", DEFAULT_DATA_DIR / "usage.jsonl")).resolve()

    max_upload_size_mb: int = _env_int("MAX_UPLOAD_SIZE_MB", 4096)
    max_upload_files: int = _env_int("MAX_UPLOAD_FILES", 20)
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".gif")
    auth_enabled: bool = _env_bool("RAG_AUTH_ENABLED", True)
    default_role: str = os.getenv("RAG_DEFAULT_ROLE", "User")
    admin_username: str = os.getenv("RAG_ADMIN_USERNAME", "admin")
    admin_password: str | None = os.getenv("RAG_ADMIN_PASSWORD", "admin")
    user_username: str = os.getenv("RAG_USER_USERNAME", "user")
    user_password: str | None = os.getenv("RAG_USER_PASSWORD", "user")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
        self.documents_path.parent.mkdir(parents=True, exist_ok=True)
        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()


def settings_for_models(
    *,
    chat_model: str | None = None,
    embedding_model: str | None = None,
    vision_model: str | None = None,
    transcription_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    vision_ingestion_enabled: bool | None = None,
    vision_detail: str | None = None,
    base_settings: Settings = settings,
) -> Settings:
    resolved_chat_model = chat_model or base_settings.openai_chat_model
    resolved_embedding_model = embedding_model or base_settings.openai_embedding_model
    resolved_vision_model = vision_model or base_settings.openai_vision_model
    resolved_transcription_model = transcription_model or base_settings.openai_transcription_model
    resolved_tts_model = tts_model or base_settings.openai_tts_model
    resolved_tts_voice = tts_voice or base_settings.openai_tts_voice
    index_dir = Path(
        os.getenv(
            "RAG_INDEX_DIR",
            base_settings.data_dir / "index" / _safe_path_part(resolved_embedding_model),
        )
    ).resolve()

    return replace(
        base_settings,
        openai_chat_model=resolved_chat_model,
        openai_embedding_model=resolved_embedding_model,
        openai_vision_model=resolved_vision_model,
        openai_transcription_model=resolved_transcription_model,
        openai_tts_model=resolved_tts_model,
        openai_tts_voice=resolved_tts_voice,
        vision_ingestion_enabled=(
            base_settings.vision_ingestion_enabled
            if vision_ingestion_enabled is None
            else vision_ingestion_enabled
        ),
        vision_detail=vision_detail or base_settings.vision_detail,
        index_dir=index_dir,
        index_path=Path(os.getenv("RAG_FAISS_INDEX_PATH", index_dir / "index.faiss")).resolve(),
        chunks_path=Path(os.getenv("RAG_CHUNKS_PATH", index_dir / "chunks.json")).resolve(),
        documents_path=Path(os.getenv("RAG_DOCUMENTS_PATH", index_dir / "documents.json")).resolve(),
    )


def configure_ssl(active_settings: Settings = settings) -> ssl.SSLContext:
    """Configure certificate defaults used by OpenAI/http clients.

    The OpenAI SDK uses httpx. We pass certificate settings directly to httpx in
    openai_client.py, and we also set SSL_CERT_FILE so lower-level libraries that
    rely on ssl defaults can find certifi or a corporate CA bundle.
    """

    if active_settings.empty_ca_bundle_fallback:
        os.environ["REQUESTS_CA_BUNDLE"] = ""
        os.environ["CURL_CA_BUNDLE"] = ""

    mode = active_settings.openai_ssl_mode.lower()
    if mode not in {"system", "certifi", "custom", "insecure"}:
        raise ValueError("OPENAI_SSL_MODE must be one of: system, certifi, custom, insecure.")

    if mode == "insecure" or not active_settings.openai_verify_ssl:
        ssl_context = ssl._create_unverified_context()
    elif mode == "system":
        ssl_context = _system_ssl_context()
    else:
        ca_bundle = certifi.where() if mode == "certifi" else active_settings.openai_ca_bundle
        os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
        ssl_context = ssl.create_default_context(cafile=ca_bundle)

    if active_settings.disable_ssl_warnings or mode == "insecure" or not active_settings.openai_verify_ssl:
        try:
            import urllib3

            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass

    return ssl_context


def _system_ssl_context() -> ssl.SSLContext:
    """Use the OS trust store, which usually includes corporate root CAs."""

    try:
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except Exception:
        return ssl.create_default_context(cafile=certifi.where())


def ssl_verify_for_httpx(active_settings: Settings = settings) -> bool | ssl.SSLContext:
    mode = active_settings.openai_ssl_mode.lower()
    if mode == "insecure" or not active_settings.openai_verify_ssl:
        return False
    return configure_ssl(active_settings)


def ssl_runtime_description(active_settings: Settings = settings) -> str:
    mode = active_settings.openai_ssl_mode.lower()
    if mode == "system":
        try:
            import truststore  # noqa: F401

            return "system (truststore)"
        except Exception:
            return "system requested; install truststore or using certifi fallback"
    if mode == "custom":
        return f"custom CA: {active_settings.openai_ca_bundle}"
    if mode == "certifi":
        return f"certifi: {certifi.where()}"
    if mode == "insecure" or not active_settings.openai_verify_ssl:
        return "insecure"
    return mode
