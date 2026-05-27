from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import re
from io import BytesIO
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import Lock
from uuid import uuid4

import pandas as pd
import plotly.express as px
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
import streamlit as st

from audio_io import response_language, synthesize_speech, transcribe_audio
from config import (
    AUDIO_TRANSCRIPTION_MODEL_OPTIONS,
    CHAT_MODEL_OPTIONS,
    EMBEDDING_MODEL_OPTIONS,
    TTS_MODEL_OPTIONS,
    TTS_VOICE_OPTIONS,
    VOICE_LANGUAGE_OPTIONS,
    VISION_MODEL_OPTIONS,
    settings,
    settings_for_models,
    ssl_runtime_description,
)
from errors import RAGApplicationError
from feedback_store import feedback_csv, feedback_jsonl, load_feedback, save_feedback
from ingestion import generate_embeddings, ingest_file, safe_filename
from rag_pipeline import RAGPipeline
from retriever import Retriever, infer_document_hashes
from usage_store import load_usage, summarize_usage, usage_csv, usage_jsonl
from vector_store import VectorStore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


PRIMARY_NAV_ITEMS = (
    "Ask",
    "Conversation",
    "Agent",
)
KNOWLEDGE_NAV_ITEMS = (
    "Dashboard",
    "Documents",
    "Retrieval Audit",
)
ADMIN_NAV_ITEMS = (
    "Ingestion",
    "Index Management",
    "Evaluation",
    "Administration",
)
NAV_GROUPS = (
    ("Primary", PRIMARY_NAV_ITEMS),
    ("Knowledge", KNOWLEDGE_NAV_ITEMS),
    ("Admin tools", ADMIN_NAV_ITEMS),
)
NAV_ITEMS = PRIMARY_NAV_ITEMS + KNOWLEDGE_NAV_ITEMS + ADMIN_NAV_ITEMS

ADMIN_ONLY_NAV = {"Ingestion", "Index Management", "Evaluation", "Administration"}
ROLES = ("Admin", "User")
NAVIGATION_MODES = ("Top row", "Sidebar", "Both")
APP_ROOT = Path(__file__).resolve().parent
RAG_VISUAL_ASSET = APP_ROOT / "docs" / "assets" / "rag-knowledge-texture.webp"
COMPACT_NAV_LABELS = {
    "Retrieval Audit": "Audit",
    "Documents": "Docs",
    "Index Management": "Index",
    "Evaluation": "Eval",
    "Administration": "Admin",
}

DEFAULT_EVALUATION_CASES = (
    {
        "Question": "Who was Heidi?",
        "Expected answer": "Heidi was a young Swiss child who lived in the Alps.",
        "Expected unknown": False,
        "Required citation contains": "Heidi",
    },
    {
        "Question": "Whom did Heidi live with?",
        "Expected answer": "Heidi lived with her grandfather, Alm-Uncle.",
        "Expected unknown": False,
        "Required citation contains": "Heidi",
    },
    {
        "Question": "Who is Black Beauty?",
        "Expected answer": "Black Beauty is a horse.",
        "Expected unknown": False,
        "Required citation contains": "Black Beauty",
    },
    {
        "Question": "What is the quarterly revenue forecast?",
        "Expected answer": "I don't know",
        "Expected unknown": True,
        "Required citation contains": "",
    },
)

DEMO_LIMIT_DEFAULTS = {
    "demo_limits_enabled": True,
    "demo_daily_call_limit": 500,
    "demo_daily_token_limit": 10_000_000,
    "demo_session_call_limit": 50,
    "demo_max_upload_files": 10,
    "demo_max_upload_size_mb": 25,
    "demo_max_top_k": 10,
    "demo_max_evaluation_cases": 20,
    "demo_max_visual_pages": 20,
    "demo_max_docx_images": 20,
}

DEMO_LIMIT_MINIMUMS = {
    "demo_daily_call_limit": 500,
    "demo_daily_token_limit": 10_000_000,
    "demo_session_call_limit": 50,
    "demo_max_upload_files": 10,
    "demo_max_upload_size_mb": 25,
    "demo_max_top_k": 10,
    "demo_max_evaluation_cases": 20,
    "demo_max_visual_pages": 20,
    "demo_max_docx_images": 20,
}


@lru_cache(maxsize=4)
def image_data_uri(path_value: str) -> str:
    path = Path(path_value)
    if not path.exists():
        return ""
    mime_type = "image/webp" if path.suffix.lower() == ".webp" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@st.cache_resource
def get_vector_store(embedding_model: str) -> VectorStore:
    return VectorStore(settings_for_models(embedding_model=embedding_model))


@st.cache_resource
def get_pipeline(chat_model: str, embedding_model: str) -> RAGPipeline:
    active_settings = settings_for_models(chat_model=chat_model, embedding_model=embedding_model)
    return RAGPipeline(get_vector_store(embedding_model), active_settings)


@st.cache_resource
def get_store_lock() -> Lock:
    return Lock()


def init_session_state() -> None:
    st.session_state.setdefault("role", settings.default_role if settings.default_role in ROLES else "Admin")
    st.session_state.setdefault("nav_selection", default_nav_selection())
    st.session_state.setdefault("chat_model", settings.openai_chat_model)
    st.session_state.setdefault("embedding_model", settings.openai_embedding_model)
    st.session_state.setdefault("vision_model", settings.openai_vision_model)
    st.session_state.setdefault("vision_ingestion_enabled", settings.vision_ingestion_enabled)
    st.session_state.setdefault("vision_detail", settings.vision_detail)
    st.session_state.setdefault("query_history", [])
    st.session_state.setdefault("conversation_messages", [])
    st.session_state.setdefault("last_ingestion", None)
    st.session_state.setdefault("authenticated", not settings.auth_enabled)
    st.session_state.setdefault("username", "")
    st.session_state.setdefault("theme_mode", "Dark")
    st.session_state.setdefault("navigation_mode", "Top row")
    st.session_state.setdefault("voice_language", settings.default_voice_language)
    st.session_state.setdefault("voice_output_enabled", settings.voice_output_enabled)
    st.session_state.setdefault("transcription_model", settings.openai_transcription_model)
    st.session_state.setdefault("tts_model", settings.openai_tts_model)
    st.session_state.setdefault("tts_voice", settings.openai_tts_voice)
    st.session_state.setdefault("speech_audio_cache", {})
    st.session_state.setdefault("last_ask_result", None)
    st.session_state.setdefault("last_agent_result", None)
    st.session_state.setdefault("agent_history", [])
    st.session_state.setdefault("evaluation_cases", load_evaluation_cases(settings))
    st.session_state.setdefault("last_evaluation_results", [])
    st.session_state.setdefault("feedback_submissions", {})
    st.session_state.setdefault("ingestion_queue", [])
    st.session_state.setdefault("demo_session_calls_used", 0)
    st.session_state.setdefault("demo_blocked_actions", [])
    st.session_state.setdefault("ask_session_id", uuid4().hex)
    st.session_state.setdefault("conversation_session_id", uuid4().hex)


def active_chat_model() -> str:
    return st.session_state.get("chat_model", settings.openai_chat_model)


def active_embedding_model() -> str:
    return st.session_state.get("embedding_model", settings.openai_embedding_model)


def active_vision_model() -> str:
    return st.session_state.get("vision_model", settings.openai_vision_model)


def active_vision_enabled() -> bool:
    return bool(st.session_state.get("vision_ingestion_enabled", settings.vision_ingestion_enabled))


def active_vision_detail() -> str:
    return st.session_state.get("vision_detail", settings.vision_detail)


def active_navigation_mode() -> str:
    mode = st.session_state.get("navigation_mode", "Top row")
    return mode if mode in NAVIGATION_MODES else "Top row"


def next_navigation_mode(mode: str) -> str:
    if mode == "Top row":
        return "Sidebar"
    if mode == "Sidebar":
        return "Both"
    return "Top row"


def active_voice_language() -> str:
    language = st.session_state.get("voice_language", settings.default_voice_language)
    return language if language in VOICE_LANGUAGE_OPTIONS else "Auto"


def active_transcription_model() -> str:
    model = st.session_state.get("transcription_model", settings.openai_transcription_model)
    return model if model in AUDIO_TRANSCRIPTION_MODEL_OPTIONS else settings.openai_transcription_model


def active_tts_model() -> str:
    model = st.session_state.get("tts_model", settings.openai_tts_model)
    return model if model in TTS_MODEL_OPTIONS else settings.openai_tts_model


def active_tts_voice() -> str:
    voice = st.session_state.get("tts_voice", settings.openai_tts_voice)
    return voice if voice in TTS_VOICE_OPTIONS else settings.openai_tts_voice


def active_settings(
    chat_model: str | None = None,
    embedding_model: str | None = None,
    vision_model: str | None = None,
    transcription_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    vision_ingestion_enabled: bool | None = None,
    vision_detail: str | None = None,
):
    return settings_for_models(
        chat_model=chat_model or active_chat_model(),
        embedding_model=embedding_model or active_embedding_model(),
        vision_model=vision_model or active_vision_model(),
        transcription_model=transcription_model or active_transcription_model(),
        tts_model=tts_model or active_tts_model(),
        tts_voice=tts_voice or active_tts_voice(),
        vision_ingestion_enabled=(
            active_vision_enabled()
            if vision_ingestion_enabled is None
            else vision_ingestion_enabled
        ),
        vision_detail=vision_detail or active_vision_detail(),
    )


def current_role() -> str:
    return st.session_state.get("role", "User")


def is_admin() -> bool:
    return current_role() == "Admin"


def can_access_nav(item: str) -> bool:
    return is_admin() or item not in ADMIN_ONLY_NAV


def accessible_nav_items() -> list[str]:
    return [item for item in NAV_ITEMS if can_access_nav(item)]


def default_nav_selection() -> str:
    items = accessible_nav_items()
    return "Ask" if "Ask" in items else (items[0] if items else "Dashboard")


def can_change_models() -> bool:
    return is_admin()


def require_admin_ui() -> bool:
    if is_admin():
        return True
    st.error("Admin role required for this workspace.")
    return False


def password_for_role(role: str) -> str | None:
    if role == "Admin":
        return settings.admin_password
    return settings.user_password


def authenticate(username: str, password: str) -> str | None:
    normalized = username.strip().lower()
    if normalized == settings.admin_username.lower() and password == settings.admin_password:
        return "Admin"
    if normalized == settings.user_username.lower() and password == settings.user_password:
        return "User"
    return None


def sign_out_current_user() -> None:
    st.session_state.authenticated = False
    st.session_state.role = "User"
    st.session_state.username = ""
    st.session_state.nav_selection = default_nav_selection()


def render_identity_controls() -> None:
    st.sidebar.markdown('<div class="sidebar-section-label">Identity</div>', unsafe_allow_html=True)

    if settings.auth_enabled:
        st.sidebar.write(f"Signed in as `{st.session_state.username or current_role().lower()}`")
        st.sidebar.write(f"Role: `{current_role()}`")
        if st.sidebar.button("Sign out", width="stretch"):
            sign_out_current_user()
            st.rerun()
        return

    role = st.sidebar.segmented_control("Session role", ROLES, default=current_role())
    if role != current_role():
        st.session_state.role = role
        if not can_access_nav(st.session_state.nav_selection):
            st.session_state.nav_selection = default_nav_selection()
        st.rerun()
    st.sidebar.caption("Local RBAC simulation. Enable RAG_AUTH_ENABLED for password-gated roles.")


def render_login_page() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 0;
            padding-bottom: 0;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .st-key-login_panel {
            width: min(520px, calc(100vw - 2rem));
            margin: 0 auto;
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(32, 38, 49, 0.98), rgba(14, 18, 25, 0.98));
            box-shadow: 0 28px 80px rgba(0, 0, 0, 0.36);
            overflow: hidden;
        }
        .st-key-login_panel .login-card {
            width: 100%;
            border: 0;
            border-radius: 0;
            box-shadow: none;
            background:
                linear-gradient(135deg, rgba(91, 140, 255, 0.15), rgba(57, 184, 200, 0.055)),
                rgba(32, 38, 49, 0.92);
            padding: 1.55rem 1.35rem 1.25rem;
            border-bottom: 1px solid var(--rag-border);
        }
        .login-eyebrow {
            color: var(--rag-cyan);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08rem;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .login-product-row {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-bottom: 0.35rem;
        }
        .st-key-login_panel .login-title {
            font-size: 1.85rem;
            line-height: 1.05;
            margin: 0;
            white-space: nowrap;
        }
        .login-badge {
            border: 1px solid rgba(91, 140, 255, 0.45);
            background: rgba(91, 140, 255, 0.14);
            color: var(--rag-text);
            border-radius: 999px;
            padding: 0.22rem 0.6rem;
            font-size: 0.78rem;
            font-weight: 800;
        }
        .st-key-login_panel .login-subtitle {
            margin: 0;
            color: var(--rag-muted);
            font-size: 0.95rem;
        }
        .st-key-login_panel [data-testid="stForm"] {
            border: 0;
            background: transparent;
            padding: 1.25rem 1.35rem 0.4rem;
        }
        .st-key-login_panel [data-testid="InputInstructions"] {
            display: none !important;
        }
        .st-key-login_panel [data-testid="stTextInput"] {
            margin-bottom: 0.8rem;
        }
        .st-key-login_panel [data-testid="stTextInput"] input {
            min-height: 3rem;
            font-size: 1rem;
            border-radius: 8px !important;
            padding-right: 3.2rem !important;
        }
        .st-key-login_panel [data-testid="stTextInput"] input[type="password"]::-ms-reveal,
        .st-key-login_panel [data-testid="stTextInput"] input[type="password"]::-ms-clear {
            display: none;
        }
        .st-key-login_panel .stButton > button {
            min-height: 3rem;
            font-size: 1rem;
            margin-top: 0.35rem;
        }
        .login-demo-note {
            color: var(--rag-muted);
            font-size: 0.88rem;
            padding: 0.15rem 1.35rem 1.35rem;
        }
        .login-demo-note strong {
            color: var(--rag-text);
            font-weight: 800;
        }
        @media (max-width: 560px) {
            .block-container {
                align-items: flex-start;
                padding-top: 2rem;
            }
            .st-key-login_panel .login-title {
                white-space: normal;
                font-size: 1.65rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1.15, 1])

    with center:
        with st.container(key="login_panel"):
            st.markdown(
                """
                <div class="login-card">
                    <div class="login-eyebrow">Knowledge workspace</div>
                    <div class="login-product-row">
                        <div class="login-title">Enterprise RAG</div>
                        <div class="login-badge">Console</div>
                    </div>
                    <div class="login-subtitle">Sign in to search documents, inspect citations, and run demos.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="admin or user")
                password = st.text_input("Password", type="password", placeholder="admin or user")
                submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

            st.markdown(
                '<div class="login-demo-note">Demo credentials: <strong>admin/admin</strong> or <strong>user/user</strong></div>',
                unsafe_allow_html=True,
            )

        if submitted:
            role = authenticate(username, password)
            if role:
                st.session_state.authenticated = True
                st.session_state.role = role
                st.session_state.username = username.strip().lower()
                st.session_state.nav_selection = default_nav_selection()
                st.rerun()
            st.error("Invalid username or password.")


def inject_enterprise_styles() -> None:
    visual_uri = image_data_uri(str(RAG_VISUAL_ASSET))
    visual_css = (
        f':root {{ --rag-visual-image: url("{visual_uri}"); }}\n'
        if visual_uri
        else ":root { --rag-visual-image: none; }\n"
    )
    base_style = """
        <style>
        :root {
            --rag-bg: #0d1117;
            --rag-bg-2: #111821;
            --rag-panel: #171b22;
            --rag-panel-2: #202631;
            --rag-panel-soft: rgba(255, 255, 255, 0.035);
            --rag-border: #303743;
            --rag-border-strong: #475263;
            --rag-text: #f6f8fb;
            --rag-muted: #aab3c2;
            --rag-muted-2: #7f8a9b;
            --rag-blue: #5b8cff;
            --rag-cyan: #39b8c8;
            --rag-green: #35c27c;
            --rag-amber: #d89b2b;
            --rag-gold: #f2bf5e;
            --rag-coral: #ff7a7a;
            --rag-red: #e46969;
            --rag-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
            --rag-shadow-soft: 0 10px 26px rgba(0, 0, 0, 0.18);
            --rag-focus: 0 0 0 2px rgba(91, 140, 255, 0.34);
        }

        .stApp {
            position: relative;
            background:
                linear-gradient(135deg, rgba(53, 194, 124, 0.06) 0%, rgba(255, 122, 122, 0.035) 42%, rgba(91, 140, 255, 0.055) 100%),
                linear-gradient(180deg, rgba(57, 184, 200, 0.055) 0%, rgba(13, 17, 23, 0) 18rem),
                linear-gradient(180deg, var(--rag-bg) 0%, #0a0d12 100%);
            color: var(--rag-text);
        }

        .stApp::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            z-index: 0;
            background-image: var(--rag-visual-image);
            background-repeat: no-repeat;
            background-position: right -9rem top 4rem;
            background-size: min(68rem, 76vw) auto;
            opacity: 0.16;
            filter: saturate(1.1);
            animation: ragAmbientDrift 22s ease-in-out infinite alternate;
        }

        .stApp > header,
        .stApp [data-testid="stAppViewContainer"] {
            position: relative;
            z-index: 1;
        }

        @keyframes ragAmbientDrift {
            from {
                transform: translate3d(0, 0, 0) scale(1);
            }
            to {
                transform: translate3d(-1.4rem, 0.8rem, 0) scale(1.025);
            }
        }

        @keyframes ragFadeLift {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes ragPulse {
            0%, 100% {
                box-shadow: 0 0 0 0 rgba(53, 194, 124, 0.22);
            }
            50% {
                box-shadow: 0 0 0 0.35rem rgba(53, 194, 124, 0);
            }
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 2.5rem;
            max-width: 1440px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        h2, h3 {
            margin-top: 0.35rem;
        }

        p, li, label, [data-testid="stMarkdownContainer"] {
            line-height: 1.5;
        }

        label,
        [data-testid="stWidgetLabel"] {
            color: var(--rag-text) !important;
            font-weight: 700 !important;
        }

        a {
            color: var(--rag-cyan);
            text-decoration-thickness: 1px;
            text-underline-offset: 0.18rem;
        }

        #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
            display: none;
        }

        header[data-testid="stHeader"] {
            height: 2.75rem;
            visibility: visible;
            background: transparent;
            z-index: 1000;
        }

        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"],
        button[title="View sidebar"],
        button[title="Hide sidebar"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            z-index: 1001;
        }

        [data-testid="stIconMaterial"],
        span[class*="material-symbols"],
        span[class*="material-icons"],
        i[class*="material-icons"] {
            display: none !important;
        }

        [data-testid="stSidebarCollapsedControl"] button::before,
        [data-testid="collapsedControl"] button::before,
        button[title="View sidebar"]::before,
        button[title="Hide sidebar"]::before {
            content: "";
            display: block;
            width: 1rem;
            height: 0.72rem;
            background:
                linear-gradient(var(--rag-text), var(--rag-text)) 0 0 / 100% 2px no-repeat,
                linear-gradient(var(--rag-text), var(--rag-text)) 0 50% / 100% 2px no-repeat,
                linear-gradient(var(--rag-text), var(--rag-text)) 0 100% / 100% 2px no-repeat;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--rag-border);
            background: linear-gradient(180deg, #121820 0%, #0d1117 100%);
        }

        .sidebar-brand {
            border: 1px solid var(--rag-border);
            background:
                linear-gradient(135deg, rgba(15, 23, 34, 0.9), rgba(15, 23, 34, 0.72)),
                var(--rag-visual-image) center / cover no-repeat;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.25rem 0 0.9rem;
            box-shadow: var(--rag-shadow-soft);
            overflow: hidden;
        }

        .sidebar-brand-title {
            color: var(--rag-text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .sidebar-brand-subtitle {
            color: var(--rag-muted);
            font-size: 0.78rem;
        }

        .login-shell {
            min-height: 82vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .login-card {
            width: min(460px, 100%);
            border: 1px solid var(--rag-border);
            background: linear-gradient(180deg, rgba(32, 38, 49, 0.98), rgba(23, 27, 34, 0.98));
            border-radius: 8px;
            padding: 1.55rem;
            box-shadow: var(--rag-shadow);
        }

        .login-title {
            color: var(--rag-text);
            font-size: 1.55rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .login-subtitle {
            color: var(--rag-muted);
            font-size: 0.9rem;
            margin-bottom: 1rem;
        }

        .topbar {
            position: relative;
            z-index: 5;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            border: 1px solid var(--rag-border);
            background: rgba(23, 29, 38, 0.94);
            backdrop-filter: blur(8px);
            border-radius: 8px;
            padding: 0.65rem 0.85rem;
            margin-bottom: 1rem;
            min-height: 3rem;
            overflow: visible;
        }

        .st-key-top_bar {
            position: relative;
            z-index: 5;
            border-color: rgba(71, 82, 99, 0.76) !important;
            background:
                linear-gradient(135deg, rgba(53, 194, 124, 0.12), rgba(255, 122, 122, 0.06) 42%, rgba(91, 140, 255, 0.1)),
                linear-gradient(180deg, rgba(28, 34, 44, 0.95), rgba(18, 23, 31, 0.95));
            backdrop-filter: blur(12px);
            border-radius: 8px;
            padding: 0.5rem 0.65rem;
            margin-bottom: 0.85rem;
            box-shadow: var(--rag-shadow-soft);
            animation: ragFadeLift 360ms ease both;
        }

        .st-key-top_bar [data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .st-key-top_bar .stButton > button {
            width: 2.5rem !important;
            min-width: 2.5rem !important;
            max-width: 2.5rem !important;
            min-height: 2.5rem;
            border-radius: 6px;
            padding-left: 0;
            padding-right: 0;
            justify-content: center;
        }

        .st-key-top_bar .stButton > button p,
        .st-key-top_bar .stButton > button [data-testid="stMarkdownContainer"],
        .st-key-navigation_mode_cycle button p,
        .st-key-navigation_mode_cycle button [data-testid="stMarkdownContainer"] {
            display: none;
        }

        .st-key-navigation_mode_cycle button {
            font-size: 0 !important;
            width: 2.5rem !important;
        }

        .st-key-navigation_mode_cycle button::before {
            content: "";
            display: block;
            width: 1.05rem;
            height: 0.72rem;
            background:
                linear-gradient(var(--rag-text), var(--rag-text)) 0 0 / 100% 2px no-repeat,
                linear-gradient(var(--rag-text), var(--rag-text)) 0 50% / 100% 2px no-repeat,
                linear-gradient(var(--rag-text), var(--rag-text)) 0 100% / 100% 2px no-repeat;
        }

        .st-key-navigation_mode_cycle button span {
            font-size: 1.1rem !important;
        }

        .st-key-top_bar [data-testid="stSelectbox"] {
            min-width: 13rem;
            max-width: 19rem;
        }

        .st-key-top_bar [data-testid="stSelectbox"] label {
            display: none;
        }

        .st-key-top_bar [data-baseweb="select"] > div {
            min-height: 2.5rem;
            border-radius: 8px;
            background: rgba(28, 36, 48, 0.92) !important;
            border-color: var(--rag-border);
            font-size: 0.82rem;
            font-weight: 700;
            position: relative;
            padding-right: 0.25rem;
        }

        [data-testid="stExpander"] details > summary {
            position: relative;
            padding-left: 2.25rem !important;
        }

        [data-testid="stExpander"] details > summary::before {
            content: "";
            position: absolute;
            left: 0.85rem;
            top: 50%;
            width: 0;
            height: 0;
            border-top: 0.28rem solid transparent;
            border-bottom: 0.28rem solid transparent;
            border-left: 0.42rem solid var(--rag-muted);
            transform: translateY(-50%);
            pointer-events: none;
        }

        [data-testid="stExpander"] details[open] > summary::before {
            border-left: 0.28rem solid transparent;
            border-right: 0.28rem solid transparent;
            border-top: 0.42rem solid var(--rag-muted);
            border-bottom: 0;
        }

        .breadcrumb {
            color: var(--rag-muted);
            font-size: 0.85rem;
            font-weight: 600;
            line-height: 1.25;
            display: flex;
            align-items: center;
            min-height: 2.5rem;
        }

        .topbar-actions {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.4rem;
            flex-wrap: wrap;
            min-height: 2.5rem;
        }

        .workspace-nav-shell {
            margin: 0.65rem 0 1.5rem 0;
            padding: 0.55rem;
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            background: rgba(23, 29, 38, 0.72);
        }

        .workspace-nav-shell .stButton > button {
            min-height: 2.3rem;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 700;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        .st-key-workspace_nav {
            margin-bottom: 1.35rem;
            padding: 0.48rem 0.6rem;
            border-color: rgba(71, 82, 99, 0.7) !important;
            background:
                linear-gradient(135deg, rgba(91, 140, 255, 0.08), rgba(53, 194, 124, 0.045), rgba(255, 122, 122, 0.035)),
                rgba(23, 27, 34, 0.82);
            border-radius: 8px;
            box-shadow: var(--rag-shadow-soft);
            animation: ragFadeLift 380ms ease both;
        }

        .st-key-workspace_nav [data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .st-key-workspace_nav button,
        .st-key-workspace_nav button p {
            white-space: nowrap;
            word-break: keep-all;
        }

        .st-key-workspace_nav [data-testid="stSegmentedControl"] {
            width: 100%;
        }

        [data-testid="stSegmentedControl"] button {
            border-radius: 6px !important;
            min-height: 2.35rem;
            font-weight: 700;
        }

        .role-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--rag-border);
            background: linear-gradient(135deg, rgba(91, 140, 255, 0.18), rgba(53, 194, 124, 0.12));
            color: var(--rag-text);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.25;
            min-height: 2rem;
        }

        .sidebar-section-label {
            color: var(--rag-muted);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04rem;
            text-transform: uppercase;
            margin: 0.75rem 0 0.35rem;
        }

        [data-testid="stSidebar"] .stButton > button {
            justify-content: flex-start;
            border-radius: 6px;
            min-height: 2.45rem;
            border: 1px solid transparent;
            font-weight: 600;
            transition: background 140ms ease, border-color 140ms ease, color 140ms ease, transform 140ms ease;
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent;
            color: var(--rag-muted);
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            color: var(--rag-text);
            background: rgba(255, 255, 255, 0.055);
            border-color: var(--rag-border);
            transform: translateX(1px);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(91, 140, 255, 0.18);
            border-color: rgba(91, 140, 255, 0.48);
            color: var(--rag-text);
        }

        .st-key-app_sidebar {
            border-color: var(--rag-border) !important;
            background: rgba(23, 29, 38, 0.72);
            border-radius: 8px;
            padding: 0.9rem;
        }

        .st-key-app_sidebar .stButton > button {
            justify-content: flex-start;
            min-height: 2.35rem;
            border-radius: 6px;
            font-weight: 700;
        }

        .rag-title {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(71, 82, 99, 0.82);
            border-radius: 8px;
            padding: 1.35rem 1.45rem;
            margin-bottom: 1.2rem;
            background:
                linear-gradient(90deg, rgba(13, 17, 23, 0.96) 0%, rgba(13, 17, 23, 0.82) 48%, rgba(13, 17, 23, 0.62) 100%),
                var(--rag-visual-image) right center / min(43rem, 48vw) auto no-repeat;
            box-shadow: var(--rag-shadow-soft);
            animation: ragFadeLift 420ms ease both;
        }

        .rag-title::before {
            content: "";
            position: absolute;
            inset: 0 0 auto;
            height: 3px;
            background: linear-gradient(90deg, var(--rag-cyan), var(--rag-green), var(--rag-gold), var(--rag-coral));
        }

        .rag-title > div {
            position: relative;
            z-index: 1;
        }

        .rag-title h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .rag-subtle {
            color: var(--rag-muted);
            font-size: 0.9rem;
            max-width: 44rem;
        }

        .rag-title-status {
            display: inline-flex;
            align-items: center;
            white-space: nowrap;
            border: 1px solid rgba(255, 255, 255, 0.12);
            background: rgba(17, 24, 33, 0.72);
            color: var(--rag-text);
            border-radius: 999px;
            padding: 0.42rem 0.7rem;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.16);
            backdrop-filter: blur(10px);
        }

        .status-dot {
            display: inline-block;
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 999px;
            background: var(--rag-green);
            margin-right: 0.4rem;
            animation: ragPulse 2.4s ease-in-out infinite;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-bottom: 1.1rem;
        }

        .metric-card {
            border: 1px solid var(--rag-border);
            background:
                linear-gradient(135deg, rgba(91, 140, 255, 0.12), rgba(53, 194, 124, 0.055) 48%, rgba(242, 191, 94, 0.05)),
                linear-gradient(180deg, rgba(32, 38, 49, 0.88), rgba(23, 27, 34, 0.96));
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: var(--rag-shadow-soft);
            position: relative;
            overflow: hidden;
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(57, 184, 200, 0.48);
            box-shadow: var(--rag-shadow);
        }

        .metric-card::before {
            content: "";
            display: block;
            width: 2.2rem;
            height: 2px;
            border-radius: 999px;
            background: var(--rag-cyan);
            margin-bottom: 0.55rem;
        }

        .metric-label {
            color: var(--rag-muted);
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--rag-text);
            font-size: 1.45rem;
            font-weight: 700;
        }

        .metric-note {
            color: var(--rag-muted);
            font-size: 0.78rem;
            margin-top: 0.25rem;
        }

        .section-panel {
            border: 1px solid var(--rag-border);
            background:
                linear-gradient(135deg, rgba(57, 184, 200, 0.06), rgba(255, 122, 122, 0.035)),
                rgba(23, 27, 34, 0.78);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0 1rem;
            box-shadow: var(--rag-shadow-soft);
        }

        .source-panel {
            border-left: 3px solid var(--rag-blue);
            background: rgba(79, 140, 255, 0.08);
            padding: 0.75rem 0.9rem;
            border-radius: 6px;
            margin-bottom: 0.55rem;
        }

        .empty-state-panel {
            border: 1px dashed var(--rag-border);
            background:
                linear-gradient(90deg, rgba(13, 17, 23, 0.88), rgba(13, 17, 23, 0.72)),
                var(--rag-visual-image) right center / min(38rem, 58vw) auto no-repeat;
            border-radius: 8px;
            padding: 1.35rem;
            margin: 0.75rem 0 1rem;
            box-shadow: var(--rag-shadow-soft);
        }

        .empty-state-title {
            color: var(--rag-text);
            font-size: 1.05rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .empty-state-copy {
            color: var(--rag-muted);
            font-size: 0.9rem;
            line-height: 1.45;
        }

        .answer-quality {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.7rem;
            border: 1px solid var(--rag-border);
            background: linear-gradient(135deg, rgba(53, 194, 124, 0.08), rgba(91, 140, 255, 0.06)), rgba(23, 27, 34, 0.78);
            border-radius: 8px;
            padding: 0.65rem 0.75rem;
            margin: 0.75rem 0 0.4rem;
            color: var(--rag-muted);
            font-size: 0.84rem;
        }

        .confidence-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            border-radius: 999px;
            padding: 0.22rem 0.58rem;
            font-weight: 800;
            color: var(--rag-text);
            border: 1px solid var(--rag-border);
        }

        .confidence-high {
            background: rgba(47, 191, 113, 0.16);
            border-color: rgba(47, 191, 113, 0.44);
        }

        .confidence-medium {
            background: rgba(216, 155, 43, 0.16);
            border-color: rgba(216, 155, 43, 0.44);
        }

        .confidence-low {
            background: rgba(224, 95, 95, 0.16);
            border-color: rgba(224, 95, 95, 0.44);
        }

        .evidence-summary {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.5rem 0 0.9rem;
        }

        .evidence-chip {
            border: 1px solid var(--rag-border);
            background: rgba(23, 29, 38, 0.64);
            border-radius: 8px;
            padding: 0.65rem;
        }

        .evidence-chip-label {
            color: var(--rag-muted);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.03rem;
        }

        .evidence-chip-value {
            color: var(--rag-text);
            font-size: 1rem;
            font-weight: 800;
            margin-top: 0.15rem;
        }

        .source-card {
            border: 1px solid var(--rag-border);
            background: linear-gradient(135deg, rgba(91, 140, 255, 0.08), rgba(242, 191, 94, 0.045)), rgba(23, 29, 38, 0.66);
            border-radius: 8px;
            padding: 0.85rem;
            margin: 0.65rem 0 0.35rem;
            transition: transform 160ms ease, border-color 160ms ease, background 160ms ease;
        }

        .source-card:hover {
            transform: translateY(-1px);
            border-color: rgba(242, 191, 94, 0.38);
        }

        .source-card-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 0.75rem;
            margin-bottom: 0.45rem;
        }

        .source-card-title {
            color: var(--rag-text);
            font-weight: 800;
            line-height: 1.3;
        }

        .source-card-meta {
            color: var(--rag-muted);
            font-size: 0.78rem;
            margin-top: 0.15rem;
        }

        .source-card-excerpt {
            color: var(--rag-text);
            font-size: 0.9rem;
            line-height: 1.5;
            overflow-wrap: anywhere;
        }

        .agent-tool-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.6rem 0 1rem;
        }

        .agent-tool-card {
            border: 1px solid var(--rag-border);
            background: rgba(23, 27, 34, 0.7);
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 5.1rem;
            transition: border-color 140ms ease, background 140ms ease, transform 140ms ease;
        }

        .agent-tool-card:hover {
            border-color: rgba(91, 140, 255, 0.42);
            background: rgba(32, 38, 49, 0.86);
            transform: translateY(-1px);
        }

        .agent-tool-title {
            color: var(--rag-text);
            font-size: 0.9rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .agent-tool-copy {
            color: var(--rag-muted);
            font-size: 0.78rem;
            line-height: 1.4;
        }

        .agent-plan {
            border-left: 3px solid var(--rag-green);
            background: rgba(47, 191, 113, 0.1);
            border-radius: 6px;
            padding: 0.75rem 0.9rem;
            margin: 0.75rem 0;
        }

        .st-key-conversation_chat_shell,
        .st-key-ask_chat_shell {
            max-width: 980px;
            margin: 0 auto;
        }

        .conversation-action-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin: 0.2rem 0 0.8rem;
        }

        .conversation-action-title {
            color: var(--rag-text);
            font-size: 0.95rem;
            font-weight: 800;
        }

        .conversation-action-meta {
            color: var(--rag-muted);
            font-size: 0.78rem;
            margin-top: 0.12rem;
        }

        .conversation-empty-state {
            min-height: 18rem;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: var(--rag-muted);
            border: 1px dashed var(--rag-border);
            border-radius: 8px;
            background:
                linear-gradient(180deg, rgba(13, 17, 23, 0.86), rgba(13, 17, 23, 0.7)),
                var(--rag-visual-image) center / cover no-repeat;
            padding: 2rem;
            margin: 1rem 0;
            box-shadow: var(--rag-shadow-soft);
        }

        .conversation-empty-title {
            color: var(--rag-text);
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .st-key-conversation_settings,
        .st-key-ask_settings,
        .st-key-agent_settings {
            border-color: var(--rag-border) !important;
            background:
                linear-gradient(135deg, rgba(57, 184, 200, 0.07), rgba(242, 191, 94, 0.035)),
                rgba(23, 27, 34, 0.8);
            border-radius: 8px;
            margin-bottom: 0.9rem;
            box-shadow: var(--rag-shadow-soft);
            animation: ragFadeLift 460ms ease both;
        }

        .st-key-conversation_chat_shell [data-testid="stChatMessage"],
        .st-key-ask_chat_shell [data-testid="stChatMessage"] {
            border-bottom: 0;
            padding: 0.7rem 0.25rem;
            background: transparent;
        }

        .st-key-conversation_chat_shell [data-testid="stChatMessage"]:last-of-type,
        .st-key-ask_chat_shell [data-testid="stChatMessage"]:last-of-type {
            border-bottom: 0;
        }

        .st-key-conversation_chat_shell [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
        .st-key-ask_chat_shell [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            line-height: 1.55;
            max-width: 820px;
        }

        [data-testid="stChatMessage"] {
            border-radius: 8px;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(91, 140, 255, 0.055);
            border: 1px solid rgba(91, 140, 255, 0.12);
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            background: rgba(255, 255, 255, 0.026);
            border: 1px solid rgba(255, 255, 255, 0.055);
        }

        .st-key-conversation_chat_shell [data-testid="stChatMessage"] [data-testid="stExpander"],
        .st-key-ask_chat_shell [data-testid="stChatMessage"] [data-testid="stExpander"] {
            margin-top: 0.45rem;
        }

        .st-key-ask_new_chat button,
        .st-key-conversation_new_chat button {
            min-width: 6.5rem;
            white-space: nowrap;
            justify-content: center;
        }

        .st-key-ask_new_chat button p,
        .st-key-conversation_new_chat button p {
            white-space: nowrap;
        }

        .source-meta {
            color: var(--rag-muted);
            font-size: 0.8rem;
            margin-bottom: 0.35rem;
        }

        .ingestion-steps {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.25rem 0 1rem;
        }

        .ingestion-step {
            border: 1px solid var(--rag-border);
            background: rgba(23, 27, 34, 0.68);
            border-radius: 8px;
            padding: 0.7rem;
            min-height: 4rem;
        }

        .ingestion-step-active {
            border-color: rgba(79, 140, 255, 0.62);
            background: rgba(79, 140, 255, 0.12);
        }

        .ingestion-step-done {
            border-color: rgba(47, 191, 113, 0.45);
            background: rgba(47, 191, 113, 0.1);
        }

        .ingestion-step-label {
            color: var(--rag-text);
            font-weight: 800;
            font-size: 0.84rem;
            margin-bottom: 0.2rem;
        }

        .ingestion-step-note {
            color: var(--rag-muted);
            font-size: 0.76rem;
            line-height: 1.35;
        }

        .audit-guide-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.5rem 0 1rem;
        }

        .audit-guide-card {
            border: 1px solid var(--rag-border);
            background: rgba(23, 27, 34, 0.68);
            border-radius: 8px;
            padding: 0.75rem;
        }

        .audit-guide-title {
            color: var(--rag-text);
            font-size: 0.9rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }

        .audit-guide-copy {
            color: var(--rag-muted);
            font-size: 0.8rem;
            line-height: 1.4;
        }

        .small-pill {
            display: inline-block;
            border: 1px solid var(--rag-border);
            background: rgba(32, 38, 49, 0.86);
            color: var(--rag-text);
            border-radius: 999px;
            padding: 0.2rem 0.5rem;
            font-size: 0.78rem;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }

        .stButton > button {
            border-radius: 6px;
            min-height: 2.4rem;
            font-weight: 700;
            transition: background 160ms ease, border-color 160ms ease, color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f86ff, #32bca7 58%, #f0b85b) !important;
            border-color: transparent !important;
            color: #ffffff !important;
            box-shadow: 0 12px 28px rgba(50, 188, 167, 0.22);
        }

        .stButton > button[kind="primary"]:hover {
            box-shadow: 0 16px 34px rgba(80, 139, 255, 0.26);
        }

        .stButton > button:focus-visible,
        [data-baseweb="select"] > div:focus-within,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stNumberInput"] input:focus {
            box-shadow: var(--rag-focus) !important;
            border-color: rgba(91, 140, 255, 0.72) !important;
        }

        [data-testid="stTextInput"] input,
        [data-testid="stTextArea"] textarea,
        [data-testid="stNumberInput"] input,
        [data-testid="stDateInput"] input,
        [data-baseweb="select"] > div {
            background: rgba(17, 24, 33, 0.86) !important;
            border-color: var(--rag-border) !important;
            border-radius: 8px !important;
        }

        [data-testid="stTextArea"] textarea {
            min-height: 6rem;
        }

        [data-testid="stFileUploader"] {
            border: 1px dashed var(--rag-border);
            border-radius: 8px;
            background: rgba(23, 27, 34, 0.54);
            padding: 0.65rem;
        }

        [data-testid="stExpander"] {
            border-color: var(--rag-border) !important;
            border-radius: 8px !important;
            background: rgba(23, 27, 34, 0.58);
        }

        [data-testid="stExpander"] summary {
            font-weight: 800;
        }

        [data-testid="stTabs"] button {
            font-weight: 800;
            color: var(--rag-muted);
        }

        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--rag-text);
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            border: 1px solid var(--rag-border);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(23, 27, 34, 0.62);
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid var(--rag-border);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--rag-border);
            background: linear-gradient(180deg, rgba(32, 38, 49, 0.82), rgba(23, 27, 34, 0.96));
            border-radius: 8px;
            padding: 0.75rem;
            box-shadow: var(--rag-shadow-soft);
        }

        div[data-testid="stMetric"] label {
            color: var(--rag-muted) !important;
        }

        code,
        pre {
            border-radius: 8px !important;
            border-color: var(--rag-border) !important;
        }

        hr {
            border-color: var(--rag-border);
        }

        @media (prefers-reduced-motion: reduce) {
            .stApp::before,
            .st-key-top_bar,
            .st-key-workspace_nav,
            .rag-title,
            .status-dot,
            .st-key-conversation_settings,
            .st-key-ask_settings,
            .st-key-agent_settings {
                animation: none !important;
            }
            .stButton > button,
            .metric-card,
            .source-card {
                transition: none !important;
            }
        }

        @media (max-width: 900px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .metric-row,
            .evidence-summary,
            .audit-guide-grid,
            .agent-tool-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .ingestion-steps {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 560px) {
            .metric-row,
            .evidence-summary,
            .audit-guide-grid,
            .agent-tool-grid,
            .ingestion-steps {
                grid-template-columns: 1fr;
            }
            .rag-title {
                align-items: flex-start;
                flex-direction: column;
                background-position: right -10rem center;
                background-size: 34rem auto;
                padding: 1.1rem;
            }
            .answer-quality,
            .source-card-header {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
    """
    st.markdown(base_style.replace("<style>", f"<style>\n{visual_css}", 1), unsafe_allow_html=True)

    if st.session_state.get("theme_mode") == "Light":
        st.markdown(
            """
            <style>
            :root {
                --rag-bg: #f6f8fb;
                --rag-bg-2: #eef3f8;
                --rag-panel: #ffffff;
                --rag-panel-2: #eef2f7;
                --rag-panel-soft: rgba(23, 32, 44, 0.035);
                --rag-border: #d7dee9;
                --rag-border-strong: #aeb9c8;
                --rag-text: #17202c;
                --rag-muted: #617086;
                --rag-muted-2: #7b8798;
                --rag-blue: #245fd6;
                --rag-cyan: #127d8d;
                --rag-green: #158554;
                --rag-amber: #9a6a09;
                --rag-gold: #b2740a;
                --rag-coral: #c7445b;
                --rag-red: #b42323;
                --rag-shadow: 0 18px 42px rgba(19, 34, 56, 0.12);
                --rag-shadow-soft: 0 10px 24px rgba(19, 34, 56, 0.08);
                --rag-focus: 0 0 0 2px rgba(36, 95, 214, 0.2);
            }
            .stApp {
                background:
                    linear-gradient(135deg, rgba(21, 133, 84, 0.055), rgba(199, 68, 91, 0.04), rgba(36, 95, 214, 0.05)),
                    linear-gradient(180deg, rgba(18, 125, 141, 0.055) 0%, rgba(246, 248, 251, 0) 18rem),
                    linear-gradient(180deg, #f6f8fb 0%, #eef3f8 100%);
            }
            .stApp::before {
                opacity: 0.1;
                filter: saturate(0.92);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f4f7fb 100%);
            }
            .topbar {
                background: rgba(255, 255, 255, 0.94);
            }
            .st-key-top_bar,
            .st-key-workspace_nav,
            .st-key-app_sidebar {
                background: linear-gradient(135deg, rgba(18, 125, 141, 0.06), rgba(178, 116, 10, 0.045)), rgba(255, 255, 255, 0.94);
            }
            .rag-title {
                background:
                    linear-gradient(90deg, rgba(255, 255, 255, 0.96) 0%, rgba(255, 255, 255, 0.88) 52%, rgba(255, 255, 255, 0.72) 100%),
                    var(--rag-visual-image) right center / min(43rem, 48vw) auto no-repeat;
            }
            .rag-title-status {
                background: rgba(255, 255, 255, 0.72);
            }
            .empty-state-panel,
            .answer-quality,
            .evidence-chip,
            .source-card,
            .ingestion-step,
            .audit-guide-card,
            .agent-tool-card,
            [data-testid="stExpander"],
            [data-testid="stFileUploader"] {
                background: rgba(255, 255, 255, 0.86);
            }
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            [data-testid="stNumberInput"] input,
            [data-testid="stDateInput"] input,
            [data-baseweb="select"] > div {
                background: #ffffff !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def render_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="rag-title">
            <div>
                <h1>{escape_html(title)}</h1>
                <div class="rag-subtle">{escape_html(subtitle)}</div>
            </div>
            <div class="rag-subtle rag-title-status"><span class="status-dot"></span>Local index online</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_size(megabytes: int) -> str:
    if megabytes >= 1024:
        return f"{megabytes / 1024:g} GB"
    return f"{megabytes} MB"


def format_timestamp(value: str | None) -> str:
    if not value:
        return "Not available"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return value


def escape_html(value: object) -> str:
    return html.escape(str(value or ""), quote=True).replace("\n", "<br>")


def compact_error_detail(exc: BaseException, max_length: int = 280) -> str:
    message = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
    message = re.sub(r"<[^>]+>", " ", str(message))
    message = html.unescape(message)
    message = re.sub(r"\s+", " ", message).strip()
    if len(message) <= max_length:
        return message
    return message[: max_length - 3] + "..."


def demo_setting(active_settings, name: str):
    value = getattr(active_settings, name, getattr(settings, name, DEMO_LIMIT_DEFAULTS[name]))
    if name not in DEMO_LIMIT_MINIMUMS:
        return value
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return DEMO_LIMIT_MINIMUMS[name]
    if numeric_value <= 0:
        return numeric_value
    return max(numeric_value, DEMO_LIMIT_MINIMUMS[name])


def demo_limits_enabled(active_settings=settings) -> bool:
    return bool(demo_setting(active_settings, "demo_limits_enabled"))


def usage_record_day(record: dict) -> str:
    timestamp = record.get("timestamp")
    if not timestamp:
        return "unknown"
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d")
    except ValueError:
        return "unknown"


def demo_usage_status(active_settings=settings) -> dict:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_records = [record for record in load_usage(active_settings) if usage_record_day(record) == today]
    return {
        "enabled": demo_limits_enabled(active_settings),
        "daily_calls": len(today_records),
        "daily_tokens": sum(int(record.get("total_tokens") or 0) for record in today_records),
        "session_calls": int(st.session_state.get("demo_session_calls_used", 0) or 0),
        "daily_call_limit": int(demo_setting(active_settings, "demo_daily_call_limit")),
        "daily_token_limit": int(demo_setting(active_settings, "demo_daily_token_limit")),
        "session_call_limit": int(demo_setting(active_settings, "demo_session_call_limit")),
    }


def limit_text(used: int, limit: int) -> str:
    return f"{used}/unlimited" if limit <= 0 else f"{used}/{limit}"


def demo_limit_reason(action_label: str, estimated_calls: int, active_settings=settings) -> str | None:
    if not demo_limits_enabled(active_settings):
        return None

    status = demo_usage_status(active_settings)
    reasons: list[str] = []
    if status["session_call_limit"] > 0 and status["session_calls"] + estimated_calls > status["session_call_limit"]:
        reasons.append(
            f"session limit reached ({limit_text(status['session_calls'], status['session_call_limit'])} calls)"
        )
    if status["daily_call_limit"] > 0 and status["daily_calls"] + estimated_calls > status["daily_call_limit"]:
        reasons.append(
            f"daily demo limit reached ({limit_text(status['daily_calls'], status['daily_call_limit'])} calls today)"
        )
    if status["daily_token_limit"] > 0 and status["daily_tokens"] >= status["daily_token_limit"]:
        reasons.append(
            f"daily token budget reached ({limit_text(status['daily_tokens'], status['daily_token_limit'])} tokens today)"
        )

    if not reasons:
        return None
    return f"{action_label} is paused because the public demo usage limit is active: {', '.join(reasons)}."


def require_demo_budget(action_label: str, *, estimated_calls: int = 1, active_settings=settings) -> bool:
    reason = demo_limit_reason(action_label, max(1, estimated_calls), active_settings)
    if reason:
        st.warning(reason)
        st.caption("Try again later, use fewer demo actions, or ask the owner to raise/disable demo limits.")
        st.session_state.demo_blocked_actions.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "action": action_label,
                "reason": reason,
            }
        )
        return False

    if demo_limits_enabled(active_settings):
        st.session_state.demo_session_calls_used = (
            int(st.session_state.get("demo_session_calls_used", 0) or 0) + max(1, estimated_calls)
        )
    return True


def estimated_rag_calls(search_mode: str) -> int:
    return 2 if str(search_mode).lower() in {"hybrid", "semantic"} else 1


def demo_top_k_limit(active_settings=settings) -> int:
    demo_max_top_k = int(demo_setting(active_settings, "demo_max_top_k"))
    if not demo_limits_enabled(active_settings) or demo_max_top_k <= 0:
        return 20
    return max(1, min(20, demo_max_top_k))


def demo_top_k_value(active_settings=settings) -> int:
    return max(1, min(active_settings.top_k, demo_top_k_limit(active_settings)))


def normalize_demo_top_k_state(key: str, active_settings=settings) -> None:
    limit = demo_top_k_limit(active_settings)
    try:
        current = int(st.session_state.get(key, demo_top_k_value(active_settings)))
    except (TypeError, ValueError):
        current = demo_top_k_value(active_settings)
    if current > limit:
        st.session_state[key] = limit


def demo_upload_file_limit(active_settings=settings) -> int:
    demo_max_upload_files = int(demo_setting(active_settings, "demo_max_upload_files"))
    if not demo_limits_enabled(active_settings) or demo_max_upload_files <= 0:
        return active_settings.max_upload_files
    return max(1, min(active_settings.max_upload_files, demo_max_upload_files))


def demo_upload_size_limit_mb(active_settings=settings) -> int:
    demo_max_upload_size_mb = int(demo_setting(active_settings, "demo_max_upload_size_mb"))
    if not demo_limits_enabled(active_settings) or demo_max_upload_size_mb <= 0:
        return active_settings.max_upload_size_mb
    return max(1, min(active_settings.max_upload_size_mb, demo_max_upload_size_mb))


def render_demo_limit_status(active_settings=settings) -> None:
    if not demo_limits_enabled(active_settings):
        return
    status = demo_usage_status(active_settings)
    st.caption(
        "Demo limits: "
        f"session calls {limit_text(status['session_calls'], status['session_call_limit'])}, "
        f"daily calls {limit_text(status['daily_calls'], status['daily_call_limit'])}, "
        f"daily tokens {limit_text(status['daily_tokens'], status['daily_token_limit'])}."
    )


def option_index(options: tuple[str, ...], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


def model_selectbox(
    label: str,
    options: tuple[str, ...],
    value: str,
    key: str,
    *,
    disabled: bool = False,
) -> str:
    return st.selectbox(
        label,
        options,
        index=option_index(options, value),
        key=key,
        disabled=disabled,
    )


def render_voice_settings(prefix: str, *, compact: bool = False) -> dict:
    if not compact:
        st.subheader("Voice")
    language = st.selectbox(
        "Input/output language",
        VOICE_LANGUAGE_OPTIONS,
        index=option_index(VOICE_LANGUAGE_OPTIONS, active_voice_language()),
        key=f"{prefix}_voice_language",
        help="Auto detects the spoken question language. Pick Hindi or English for demo certainty.",
    )
    st.session_state.voice_language = language

    spoken_answer = st.toggle(
        "Play answer audio",
        value=bool(st.session_state.get("voice_output_enabled", settings.voice_output_enabled)),
        key=f"{prefix}_voice_output_enabled",
    )
    st.session_state.voice_output_enabled = spoken_answer

    voice = st.selectbox(
        "Answer voice",
        TTS_VOICE_OPTIONS,
        index=option_index(TTS_VOICE_OPTIONS, active_tts_voice()),
        key=f"{prefix}_tts_voice",
        disabled=not spoken_answer,
    )
    st.session_state.tts_voice = voice

    audio_model_context = st.container() if compact else st.expander("Audio models")
    with audio_model_context:
        if compact:
            model_col_a, model_col_b = st.columns(2)
            with model_col_a:
                transcription_model = st.selectbox(
                    "Speech-to-text",
                    AUDIO_TRANSCRIPTION_MODEL_OPTIONS,
                    index=option_index(AUDIO_TRANSCRIPTION_MODEL_OPTIONS, active_transcription_model()),
                    key=f"{prefix}_transcription_model",
                )
            with model_col_b:
                tts_model = st.selectbox(
                    "Text-to-speech",
                    TTS_MODEL_OPTIONS,
                    index=option_index(TTS_MODEL_OPTIONS, active_tts_model()),
                    key=f"{prefix}_tts_model",
                    disabled=not spoken_answer,
                )
        else:
            transcription_model = st.selectbox(
                "Speech-to-text",
                AUDIO_TRANSCRIPTION_MODEL_OPTIONS,
                index=option_index(AUDIO_TRANSCRIPTION_MODEL_OPTIONS, active_transcription_model()),
                key=f"{prefix}_transcription_model",
            )
            tts_model = st.selectbox(
                "Text-to-speech",
                TTS_MODEL_OPTIONS,
                index=option_index(TTS_MODEL_OPTIONS, active_tts_model()),
                key=f"{prefix}_tts_model",
                disabled=not spoken_answer,
            )
    st.session_state.transcription_model = transcription_model
    st.session_state.tts_model = tts_model
    if not compact:
        st.caption("Voice playback is AI-generated.")

    return {
        "language": language,
        "spoken_answer": spoken_answer,
        "transcription_model": transcription_model,
        "tts_model": tts_model,
        "tts_voice": voice,
    }


def audio_runtime_settings(base_settings, voice_settings: dict):
    return active_settings(
        chat_model=base_settings.openai_chat_model,
        embedding_model=base_settings.openai_embedding_model,
        vision_model=base_settings.openai_vision_model,
        transcription_model=voice_settings["transcription_model"],
        tts_model=voice_settings["tts_model"],
        tts_voice=voice_settings["tts_voice"],
        vision_ingestion_enabled=base_settings.vision_ingestion_enabled,
        vision_detail=base_settings.vision_detail,
    )


def render_voice_input(
    prefix: str,
    label: str,
    voice_settings: dict,
    runtime_settings,
    *,
    target_text_key: str | None = None,
) -> str:
    recorded_audio = st.audio_input(label, key=f"{prefix}_audio_input", sample_rate=16000)
    if recorded_audio is None:
        return st.session_state.get(f"{prefix}_transcript", "")

    audio_bytes = recorded_audio.getvalue()
    audio_hash = hashlib.sha256(audio_bytes).hexdigest()
    if st.session_state.get(f"{prefix}_audio_hash") != audio_hash:
        if not require_demo_budget("Voice transcription", active_settings=runtime_settings):
            return st.session_state.get(f"{prefix}_transcript", "")
        with st.spinner("Transcribing voice input"):
            transcript = transcribe_audio(
                audio_bytes,
                filename=f"{prefix}_voice.wav",
                language=voice_settings["language"],
                active_settings=runtime_settings,
            )
        st.session_state[f"{prefix}_audio_hash"] = audio_hash
        st.session_state[f"{prefix}_transcript"] = transcript.text
        st.session_state[f"{prefix}_transcript_language"] = transcript.language
        if target_text_key:
            st.session_state[target_text_key] = transcript.text

    transcript_text = st.session_state.get(f"{prefix}_transcript", "")
    transcript_language = st.session_state.get(f"{prefix}_transcript_language", voice_settings["language"])
    if transcript_text:
        st.caption(f"Voice transcript ({transcript_language}): {transcript_text}")
    return transcript_text


def render_spoken_answer(
    answer: str,
    voice_settings: dict,
    runtime_settings,
    *,
    language: str,
    key_prefix: str,
) -> None:
    if not voice_settings["spoken_answer"] or not answer.strip():
        return

    cache_key = hashlib.sha256(
        "|".join(
            [
                runtime_settings.openai_tts_model,
                runtime_settings.openai_tts_voice,
                language,
                answer,
            ]
        ).encode("utf-8")
    ).hexdigest()

    cache = st.session_state.speech_audio_cache
    if cache_key not in cache:
        if not require_demo_budget("Voice playback", active_settings=runtime_settings):
            return
        with st.spinner("Generating voice answer"):
            try:
                cache[cache_key] = synthesize_speech(
                    answer,
                    language=language,
                    active_settings=runtime_settings,
                )
            except RAGApplicationError as exc:
                st.warning("Voice playback unavailable on this network. Text answer is shown above.")
                with st.expander("Voice playback detail", expanded=False):
                    st.caption(compact_error_detail(exc))
                return
            except Exception as exc:
                logger.exception("Voice playback failed: %s", exc)
                st.warning("Voice playback unavailable right now. Text answer is shown above.")
                with st.expander("Voice playback detail", expanded=False):
                    st.caption(compact_error_detail(exc))
                return

    st.audio(cache[cache_key], format="audio/mp3")


def render_source_filters(prefix: str, store: VectorStore, *, use_expander: bool = True) -> dict:
    documents = store.list_documents()
    filters: dict = {}

    filter_context = st.expander("Source filters") if use_expander else st.container()
    with filter_context:
        if not documents:
            st.caption("No documents available for filtering.")
            return filters

        document_options = {document["file_hash"]: document.get("file_name", document["file_hash"]) for document in documents}
        selected_documents = st.multiselect(
            "Documents",
            options=list(document_options.keys()),
            format_func=lambda value: document_options.get(value, value),
            key=f"{prefix}_filter_documents",
        )
        if selected_documents:
            filters["document_hashes"] = selected_documents

        file_types = sorted(
            {
                Path(document.get("file_name", "")).suffix.lower().lstrip(".")
                for document in documents
                if Path(document.get("file_name", "")).suffix
            }
        )
        selected_file_types = st.multiselect(
            "File type",
            options=file_types,
            key=f"{prefix}_filter_file_types",
        )
        if selected_file_types:
            filters["file_types"] = selected_file_types

        selected_source_types = st.multiselect(
            "Source type",
            options=["text", "image"],
            key=f"{prefix}_filter_source_types",
            help="Image means visual descriptions generated from PDFs, DOCX files, or image uploads.",
        )
        if selected_source_types:
            filters["source_types"] = selected_source_types

        use_date_filter = st.toggle("Filter by upload date", value=False, key=f"{prefix}_filter_date_enabled")
        if use_date_filter:
            date_col_a, date_col_b = st.columns(2)
            with date_col_a:
                uploaded_after = st.date_input("Uploaded from", value=None, key=f"{prefix}_uploaded_after")
            with date_col_b:
                uploaded_before = st.date_input("Uploaded to", value=None, key=f"{prefix}_uploaded_before")
            if uploaded_after:
                filters["uploaded_after"] = f"{uploaded_after.isoformat()}T00:00:00+00:00"
            if uploaded_before:
                filters["uploaded_before"] = f"{uploaded_before.isoformat()}T23:59:59+00:00"

        path_query = st.text_input(
            "Folder/path contains",
            key=f"{prefix}_filter_path",
            placeholder="Example: finance, policy, uploads",
        )
        if path_query.strip():
            filters["path_query"] = path_query.strip()

        metadata_query = st.text_input(
            "Metadata/tags contain",
            key=f"{prefix}_filter_metadata",
            placeholder="File name, source type, stored tags",
        )
        if metadata_query.strip():
            filters["metadata_query"] = metadata_query.strip()

    return filters


def render_feedback_controls(
    *,
    feedback_key: str,
    query: str,
    answer: str,
    result: dict,
    runtime_settings,
    search_mode: str,
    filters: dict | None,
    context: str,
) -> None:
    st.subheader("Feedback")
    st.caption("Feedback is stored locally for Admin review and export.")
    comment = st.text_input("Optional note", key=f"{feedback_key}_comment")
    col_a, col_b = st.columns(2)
    submitted = st.session_state.feedback_submissions.get(feedback_key)

    if submitted:
        st.success(f"Feedback recorded: {submitted}")
        return

    def submit(sentiment: str) -> None:
        payload = {
            "feedback_key": feedback_key,
            "context": context,
            "sentiment": sentiment,
            "bad_retrieval": sentiment == "down",
            "comment": comment.strip(),
            "query": query,
            "answer": answer,
            "role": current_role(),
            "username": st.session_state.get("username", ""),
            "model": runtime_settings.openai_chat_model,
            "embedding_model": runtime_settings.openai_embedding_model,
            "search_mode": search_mode,
            "filters": filters or {},
            "confidence": result.get("confidence", 0.0),
            "source_count": len(result.get("sources", [])),
            "source_metadata": result.get("source_metadata", []),
            "sources": result.get("sources", []),
        }
        save_feedback(payload, runtime_settings)
        st.session_state.feedback_submissions[feedback_key] = "useful" if sentiment == "up" else "needs review"
        st.rerun()

    with col_a:
        if st.button("Good answer", key=f"{feedback_key}_up", width="stretch"):
            submit("up")
    with col_b:
        if st.button("Bad retrieval", key=f"{feedback_key}_down", width="stretch"):
            submit("down")


FOLLOW_UP_MARKERS = (
    " this ",
    " that ",
    " it ",
    " its ",
    " she ",
    " her ",
    " hers ",
    " he ",
    " him ",
    " his ",
    " they ",
    " them ",
    " their ",
    " theirs ",
    " these ",
    " those ",
    " the story ",
    " this story ",
    " the book ",
    " this book ",
    " the horse ",
    " grandfather",
    " grandmother",
    " father",
    " mother",
    " uncle",
    " aunt",
    " owner",
    " master",
)


def is_follow_up_query(query: str) -> bool:
    normalized = f" {query.strip().lower()} "
    return any(marker in normalized for marker in FOLLOW_UP_MARKERS)


def should_use_conversation_context(query: str, documents: dict[str, dict] | None = None) -> bool:
    if not is_follow_up_query(query):
        return False
    return not (documents and infer_document_hashes(query, documents))


def dominant_source_hashes(result: dict | None) -> list[str]:
    if not result:
        return []
    hashes = [
        metadata.get("file_hash")
        for metadata in result.get("source_metadata", [])
        if metadata.get("file_hash")
    ]
    if not hashes:
        return []
    counts = Counter(hashes)
    highest = max(counts.values())
    return [file_hash for file_hash, count in counts.items() if count == highest]


def latest_conversation_source_hashes() -> list[str]:
    for citations in latest_conversation_citation_groups():
        hashes = [metadata.get("file_hash") for metadata in citations if metadata.get("file_hash")]
        if hashes:
            counts = Counter(hashes)
            highest = max(counts.values())
            return [file_hash for file_hash, count in counts.items() if count == highest]
    return []


def latest_conversation_citation_groups():
    for message in reversed(st.session_state.conversation_messages):
        if message.get("role") != "assistant":
            continue
        citations = message.get("citations") or message.get("result", {}).get("source_metadata", [])
        if citations:
            yield citations


def latest_conversation_citation_chunks(store: VectorStore, limit: int = 6) -> list[dict]:
    citations = next(latest_conversation_citation_groups(), [])
    if not citations:
        return []

    lookup: dict[tuple[str, int], dict] = {}
    for record in store.chunks:
        metadata = record.get("metadata", {})
        file_hash = metadata.get("file_hash")
        chunk_index = metadata.get("chunk_index")
        if file_hash is None or chunk_index is None:
            continue
        try:
            lookup[(str(file_hash), int(chunk_index))] = record
        except (TypeError, ValueError):
            continue

    pinned: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for metadata in citations:
        file_hash = metadata.get("file_hash")
        chunk_index = metadata.get("chunk_index")
        if file_hash is None or chunk_index is None:
            continue
        try:
            key = (str(file_hash), int(chunk_index))
        except (TypeError, ValueError):
            continue
        if key in seen or key not in lookup:
            continue
        seen.add(key)
        record = lookup[key]
        pinned.append(
            {
                "text": record.get("text", ""),
                "metadata": record.get("metadata", {}),
                "score": max(0.0, min(1.0, float(metadata.get("score", 0.0) or 0.0))),
                "semantic_score": float(metadata.get("semantic_score", 0.0) or 0.0),
                "keyword_score": float(metadata.get("keyword_score", 0.0) or 0.0),
                "retrieval_method": "previous-citation",
            }
        )
        if len(pinned) >= limit:
            break
    return pinned


def reset_ask_chat() -> None:
    for key in (
        "ask_query",
        "ask_voice_review",
        "ask_transcript",
        "ask_transcript_language",
        "ask_audio_hash",
        "last_ask_result",
    ):
        st.session_state.pop(key, None)
    st.session_state.ask_session_id = uuid4().hex


def reset_conversation_chat() -> None:
    for key in (
        "conversation_voice_review",
        "conversation_transcript",
        "conversation_transcript_language",
        "conversation_audio_hash",
    ):
        st.session_state.pop(key, None)
    st.session_state.conversation_messages = []
    st.session_state.conversation_session_id = uuid4().hex


def reset_agent_workspace() -> None:
    for key in (
        "agent_goal",
        "agent_voice_review",
        "agent_transcript",
        "agent_transcript_language",
        "agent_audio_hash",
        "last_agent_result",
    ):
        st.session_state.pop(key, None)


def follow_up_filters(
    query: str,
    filters: dict | None,
    source_hashes: list[str],
    documents: dict[str, dict] | None = None,
) -> dict:
    effective_filters = dict(filters or {})
    if effective_filters.get("document_hashes") or not should_use_conversation_context(query, documents):
        return effective_filters
    if not source_hashes:
        return effective_filters
    effective_filters["document_hashes"] = source_hashes
    return effective_filters


def contextual_follow_up_query(
    query: str,
    last_result: dict | None,
    documents: dict[str, dict] | None = None,
) -> str:
    if not last_result or not should_use_conversation_context(query, documents):
        return query

    previous_query = str(last_result.get("query") or "").strip()
    previous_answer = str(last_result.get("result", {}).get("answer") or "").strip()
    if not previous_query and not previous_answer:
        return query

    return "\n".join(
        item
        for item in (
            f"Previous question: {previous_query}" if previous_query else "",
            f"Previous answer: {previous_answer}" if previous_answer else "",
            f"Current question: {query.strip()}",
        )
        if item
    )


def conversation_retrieval_query(query: str, documents: dict[str, dict] | None = None) -> str:
    if not st.session_state.conversation_messages or not should_use_conversation_context(query, documents):
        return query
    return conversation_context_prompt(query)


def render_top_bar(selected: str) -> None:
    with st.container(border=True, key="top_bar"):
        menu_col, breadcrumb_col, actions_col = st.columns([0.045, 0.655, 0.3], gap="small")
        with menu_col:
            render_navigation_menu()
        with breadcrumb_col:
            st.markdown(
                f'<div class="breadcrumb">Enterprise RAG / {escape_html(selected)}</div>',
                unsafe_allow_html=True,
            )
        with actions_col:
            role_col, model_col = st.columns([0.28, 0.72], gap="small")
            with role_col:
                st.markdown(
                    f"""
                    <div class="topbar-actions">
                        <span class="role-badge">{escape_html(current_role())}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with model_col:
                render_top_embedding_selector()


def render_top_embedding_selector() -> None:
    current_embedding = active_embedding_model()
    selected_embedding = st.selectbox(
        "Knowledge index",
        EMBEDDING_MODEL_OPTIONS,
        index=option_index(EMBEDDING_MODEL_OPTIONS, current_embedding),
        key=f"top_bar_embedding_model_{current_embedding}",
        label_visibility="collapsed",
    )
    if selected_embedding != current_embedding:
        st.session_state.embedding_model = selected_embedding
        st.rerun()


def render_navigation_menu() -> None:
    mode = active_navigation_mode()
    if st.button(
        "Navigation layout",
        key="navigation_mode_cycle",
        help=f"Current: {mode}. Click to switch to {next_navigation_mode(mode)}.",
        width="content",
    ):
        st.session_state.navigation_mode = next_navigation_mode(mode)
        st.rerun()


def render_workspace_nav(selected: str) -> None:
    items = accessible_nav_items()
    if selected not in items:
        selected = default_nav_selection()
        st.session_state.nav_selection = selected

    with st.container(border=True, key="workspace_nav"):
        nav_col, action_col = st.columns([1, 0.12], gap="small")
        with nav_col:
            chosen = st.segmented_control(
                "Workspace navigation",
                items,
                default=selected,
                format_func=lambda item: COMPACT_NAV_LABELS.get(item, item),
                key=f"workspace_nav_choice_{selected}",
                label_visibility="collapsed",
                width="stretch",
            )
            if chosen and chosen != selected:
                st.session_state.nav_selection = chosen
                st.rerun()

        if settings.auth_enabled:
            with action_col:
                if st.button("Sign out", key="workspace_sign_out", width="stretch"):
                    sign_out_current_user()
                    st.rerun()


def render_app_sidebar(selected: str) -> str:
    items = accessible_nav_items()
    if selected not in items:
        selected = default_nav_selection()
        st.session_state.nav_selection = selected

    with st.container(border=True, key="app_sidebar"):
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">Enterprise RAG Console</div>
                <div class="sidebar-brand-subtitle">Knowledge retrieval workspace</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if settings.auth_enabled:
            st.caption(f"Signed in as {st.session_state.username or current_role().lower()} - {current_role()}")
            if st.button("Sign out", key="app_sidebar_sign_out", width="stretch"):
                sign_out_current_user()
                st.rerun()
            st.divider()

        for label, group_items in NAV_GROUPS:
            available_group_items = [item for item in group_items if item in items]
            if not available_group_items:
                continue
            st.markdown(f'<div class="sidebar-section-label">{escape_html(label)}</div>', unsafe_allow_html=True)
            for item in available_group_items:
                button_type = "primary" if selected == item else "secondary"
                if st.button(
                    item,
                    key=f"app_side_nav_{item}",
                    width="stretch",
                    type=button_type,
                ):
                    st.session_state.nav_selection = item
                    st.rerun()

        st.divider()
        if not settings.auth_enabled:
            st.markdown('<div class="sidebar-section-label">Session</div>', unsafe_allow_html=True)
            st.write(f"`{current_role()}`")

        st.markdown('<div class="sidebar-section-label">Runtime</div>', unsafe_allow_html=True)
        st.caption(f"Embedding: {active_embedding_model()}")
        st.caption(f"Chat: {active_chat_model()}")
        st.caption(f"Vision: {active_vision_model()}")

        if st.button("Refresh index", key="app_side_refresh", width="stretch"):
            get_pipeline.clear()
            get_vector_store.clear()
            st.rerun()

    return st.session_state.nav_selection


def document_rows(embedding_model: str | None = None) -> list[dict]:
    active = active_settings(embedding_model=embedding_model)
    rows: list[dict] = []
    for document in get_vector_store(active.openai_embedding_model).list_documents():
        rows.append(
            {
                "Document": document.get("file_name", "Unknown"),
                "Chunks": document.get("chunk_count", 0),
                "Visual chunks": document.get("visual_chunk_count", 0),
                "Embedding model": document.get("embedding_model", active.openai_embedding_model),
                "Indexed at": format_timestamp(document.get("uploaded_at")),
            }
        )
    return sorted(rows, key=lambda item: item["Document"].lower())


def index_stats(embedding_model: str | None = None) -> dict:
    active = active_settings(embedding_model=embedding_model)
    store = get_vector_store(active.openai_embedding_model)
    documents = store.list_documents()
    latest = max((item.get("uploaded_at") for item in documents if item.get("uploaded_at")), default=None)
    return {
        "documents": len(documents),
        "chunks": store.total_vectors,
        "latest": format_timestamp(latest),
        "model": active.openai_embedding_model,
    }


def resolve_document_path(file_hash: str | None, runtime_settings) -> Path | None:
    if not file_hash:
        return None

    store = get_vector_store(runtime_settings.openai_embedding_model)
    document = store.get_document(file_hash) or {}
    source_path = document.get("source_path")
    if source_path:
        candidate = Path(source_path)
        if candidate.exists():
            return candidate

    matches = list(runtime_settings.upload_dir.glob(f"{file_hash[:12]}_*"))
    return matches[0] if matches else None


def citation_filename(metadata: dict, suffix: str = "txt") -> str:
    file_name = safe_filename(metadata.get("file_name") or "source")
    chunk = metadata.get("chunk_index", "chunk")
    return f"{Path(file_name).stem}_chunk_{chunk}.{suffix}"


def source_metadata_rows(metadata: dict) -> list[dict]:
    def display_value(value: object) -> str:
        if value is None or value == "":
            return "N/A"
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    return [
        {"Field": "Document", "Value": display_value(metadata.get("file_name") or "Unknown")},
        {"Field": "Page", "Value": display_value(metadata.get("page_number") or "N/A")},
        {"Field": "Source type", "Value": display_value(metadata.get("source_type") or "text")},
        {"Field": "Image", "Value": display_value(metadata.get("image_index") or "N/A")},
        {"Field": "Chunk", "Value": display_value(metadata.get("chunk_index"))},
        {"Field": "Similarity", "Value": display_value(metadata.get("score"))},
        {"Field": "Token start", "Value": display_value(metadata.get("token_start"))},
        {"Field": "Token count", "Value": display_value(metadata.get("token_count"))},
        {"Field": "Embedding model", "Value": display_value(metadata.get("embedding_model"))},
        {"Field": "Retrieval", "Value": display_value(metadata.get("retrieval_method") or "semantic")},
        {"Field": "Semantic score", "Value": display_value(metadata.get("semantic_score"))},
        {"Field": "Keyword score", "Value": display_value(metadata.get("keyword_score"))},
    ]


def suppress_mupdf_diagnostics(fitz_module) -> None:
    tools = getattr(fitz_module, "TOOLS", None)
    if tools is None:
        return
    for function_name in ("mupdf_display_errors", "mupdf_display_warnings"):
        function = getattr(tools, function_name, None)
        if callable(function):
            try:
                function(False)
            except Exception:
                pass


@st.cache_data(show_spinner=False)
def pdf_page_preview_bytes(path_text: str, page_number: int, modified_at: float) -> bytes:
    del modified_at
    import fitz

    suppress_mupdf_diagnostics(fitz)
    with fitz.open(path_text) as document:
        if len(document) == 0:
            raise RAGApplicationError("PDF has no pages to preview.")

        page_index = max(0, min(page_number - 1, len(document) - 1))
        page = document[page_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(1.25, 1.25), alpha=False)
        return pixmap.tobytes("png")


def render_source_preview(metadata: dict, runtime_settings) -> None:
    source_path = resolve_document_path(metadata.get("file_hash"), runtime_settings)
    if not source_path or not source_path.exists():
        st.caption("Preview unavailable. Stored source file was not found.")
        return

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        page_number = int(metadata.get("page_number") or 1)
        try:
            preview = pdf_page_preview_bytes(
                str(source_path),
                page_number,
                source_path.stat().st_mtime,
            )
            st.image(preview, caption=f"{source_path.name} - page {page_number}", width="stretch")
        except Exception as exc:
            st.caption(f"PDF preview unavailable: {exc}")
        return

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        st.image(str(source_path), caption=source_path.name, width="stretch")
        return

    if suffix == ".txt":
        try:
            st.text(source_path.read_text(encoding="utf-8", errors="replace")[:3000])
        except Exception as exc:
            st.caption(f"Text preview unavailable: {exc}")
        return

    st.caption("Preview is available for PDF, image, and text sources. Download or open this document for full review.")


def confidence_status(result: dict) -> tuple[str, str, str]:
    score = float(result.get("confidence", 0.0) or 0.0)
    source_count = len(result.get("sources", []))
    if source_count == 0 or score <= 0:
        return "No evidence", "low", "No source chunks passed the current retrieval filters."
    if score >= 0.72 and source_count >= 2:
        return "High confidence", "high", "Strong retrieved evidence with multiple supporting chunks."
    if score >= 0.45:
        return "Medium confidence", "medium", "Usable retrieved evidence; review citations for important decisions."
    return "Low confidence", "low", "Weak retrieved evidence; adjust filters or ask a more specific question."


def render_answer_quality(result: dict, min_score: float) -> None:
    label, tone, note = confidence_status(result)
    source_count = len(result.get("sources", []))
    score = float(result.get("confidence", 0.0) or 0.0)
    st.markdown(
        f"""
        <div class="answer-quality">
            <span class="confidence-badge confidence-{tone}">{escape_html(label)}</span>
            <span>{escape_html(note)} Sources: {source_count}. Score: {score:.2f}. Threshold: {min_score:.2f}.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_index_empty_state(context: str, key_prefix: str) -> None:
    admin_copy = "Upload documents to create the searchable index for this workspace."
    user_copy = "No documents are indexed yet. Ask an admin to upload documents before using this workspace."
    st.markdown(
        f"""
        <div class="empty-state-panel">
            <div class="empty-state-title">No indexed documents available</div>
            <div class="empty-state-copy">{escape_html(admin_copy if is_admin() else user_copy)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if is_admin():
        if st.button("Go to Ingestion", key=f"{key_prefix}_go_to_ingestion", type="primary"):
            st.session_state.nav_selection = "Ingestion"
            st.rerun()
    else:
        st.caption(f"{context} will be available after documents are indexed.")


def render_ingestion_steps(active_step: str = "upload") -> None:
    steps = [
        ("upload", "Upload", "Files enter the queue"),
        ("extract", "Extract", "Text and visuals are read"),
        ("chunk", "Chunk", "Content is split by tokens"),
        ("embed", "Embed", "Chunks become vectors"),
        ("index", "Index", "FAISS and metadata are saved"),
    ]
    order = [step[0] for step in steps]
    active_position = order.index(active_step) if active_step in order else 0
    cards = []
    for position, (key, label, note) in enumerate(steps):
        state_class = "ingestion-step-active" if position == active_position else ""
        if position < active_position:
            state_class = "ingestion-step-done"
        cards.append(
            f'<div class="ingestion-step {state_class}">'
            f'<div class="ingestion-step-label">{escape_html(label)}</div>'
            f'<div class="ingestion-step-note">{escape_html(note)}</div>'
            "</div>"
        )
    st.markdown(f'<div class="ingestion-steps">{"".join(cards)}</div>', unsafe_allow_html=True)


def retrieval_reason(method: str | None) -> str:
    normalized = (method or "semantic").strip().lower()
    if normalized == "keyword":
        return "Selected because exact query terms matched the chunk text or metadata."
    if normalized == "hybrid":
        return "Selected because semantic similarity and keyword evidence both contributed to its rank."
    if normalized == "context-window":
        return "Included as neighboring context around a stronger retrieved match."
    if normalized == "document-overview":
        return "Included to provide broad document context for an overview-style question."
    if normalized == "document-selection":
        return "Included because the user selected this document as the agent's focus."
    return "Selected because the query embedding is close to this chunk embedding."


AGENT_TOOL_LABELS = {
    "search_documents": "Search Documents",
    "summarize_documents": "Summarize Documents",
    "compare_documents": "Compare Documents",
    "generate_report": "Generate Report",
}

AGENT_DOCUMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "answer",
    "brief",
    "citation",
    "citations",
    "compare",
    "create",
    "difference",
    "differences",
    "doc",
    "docs",
    "document",
    "documents",
    "file",
    "files",
    "generate",
    "give",
    "key",
    "make",
    "of",
    "on",
    "pdf",
    "report",
    "summarize",
    "summary",
    "the",
    "to",
    "txt",
    "with",
}


def choose_agent_tool(goal: str, requested_tool: str, selected_hashes: list[str]) -> str:
    if requested_tool != "Auto":
        return requested_tool

    lowered = goal.lower()
    wants_compare = any(term in lowered for term in ("compare", "difference", "versus", " vs "))
    wants_report = any(term in lowered for term in ("report", "brief", "write-up", "writeup", "executive summary"))
    wants_summary = any(term in lowered for term in ("summarize", "summary", "overview", "key points", "main points"))

    if wants_compare and len(selected_hashes) >= 2:
        return "compare_documents"
    if wants_report:
        return "generate_report"
    if wants_summary:
        return "summarize_documents"
    if wants_compare:
        return "search_documents"
    return "search_documents"


def agent_document_terms(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower(), flags=re.IGNORECASE)
        if token not in AGENT_DOCUMENT_STOPWORDS and len(token) > 1 and not token.isdigit()
    ]


def infer_agent_document_hashes(goal: str, documents: dict[str, dict]) -> list[str]:
    query_terms = set(agent_document_terms(goal))
    if not query_terms:
        return []

    matches: list[tuple[int, str]] = []
    for file_hash, document in documents.items():
        title_terms = set(agent_document_terms(Path(document.get("file_name", "")).stem))
        if not title_terms:
            continue
        overlap = len(query_terms & title_terms)
        if overlap:
            matches.append((overlap, file_hash))

    return [file_hash for _, file_hash in sorted(matches, key=lambda item: item[0], reverse=True)]


def unique_hashes(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value and value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def describe_documents(file_hashes: list[str], documents_by_hash: dict[str, dict]) -> str:
    names = []
    for file_hash in file_hashes:
        document = documents_by_hash.get(file_hash, {})
        if isinstance(document, dict):
            names.append(document.get("file_name", file_hash))
        else:
            names.append(str(document or file_hash))
    return ", ".join(names) if names else "None"


def selected_document_filters(selected_hashes: list[str]) -> dict | None:
    return {"document_hashes": selected_hashes} if selected_hashes else None


def collect_document_context_chunks(
    store: VectorStore,
    selected_hashes: list[str],
    *,
    limit_per_document: int = 8,
) -> list[dict]:
    allowed = set(selected_hashes)
    grouped: dict[str, list[dict]] = {}
    for record in store.chunks:
        metadata = record.get("metadata", {})
        file_hash = metadata.get("file_hash")
        if allowed and file_hash not in allowed:
            continue
        grouped.setdefault(str(file_hash), []).append(record)

    chunks: list[dict] = []
    for records in grouped.values():
        for record in records[:limit_per_document]:
            chunks.append(
                {
                    "text": record.get("text", ""),
                    "metadata": record.get("metadata", {}),
                    "score": 0.8,
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "retrieval_method": "document-selection",
                }
            )
    return chunks


def agent_prompt_for_tool(tool_name: str, goal: str, selected_documents: list[str]) -> str:
    selected_text = ", ".join(selected_documents) if selected_documents else "the retrieved documents"
    if tool_name == "summarize_documents":
        return (
            f"Summarize {selected_text} for the user goal below. "
            "Return concise sections: overview, key facts, risks or gaps, and useful citations.\n\n"
            f"Goal: {goal}"
        )
    if tool_name == "compare_documents":
        return (
            f"Compare {selected_text} for the user goal below. "
            "Return a structured comparison with similarities, differences, contradictions, and citation-backed notes.\n\n"
            f"Goal: {goal}"
        )
    if tool_name == "generate_report":
        return (
            f"Create a professional Markdown report for {selected_text}. "
            "Use headings, concise bullets, key evidence, risks, recommended next steps, and cite the provided sources.\n\n"
            f"Goal: {goal}"
        )
    return goal


def run_agentic_rag(
    *,
    goal: str,
    requested_tool: str,
    chat_model: str,
    embedding_model: str,
    selected_hashes: list[str],
    top_k: int,
    min_score: float,
    search_mode: str,
    response_language_name: str,
) -> dict:
    runtime_settings = active_settings(chat_model=chat_model, embedding_model=embedding_model)
    store = get_vector_store(embedding_model)
    pipeline = get_pipeline(chat_model, embedding_model)
    documents_by_hash = {document["file_hash"]: document for document in store.list_documents()}
    inferred_hashes = infer_agent_document_hashes(goal, documents_by_hash)
    if not inferred_hashes:
        inferred_hashes = infer_document_hashes(goal, documents_by_hash)
    effective_hashes = unique_hashes(selected_hashes + inferred_hashes)
    selected_names = [
        documents_by_hash.get(file_hash, {}).get("file_name", file_hash) for file_hash in effective_hashes
    ]
    tool_name = choose_agent_tool(goal, requested_tool, effective_hashes)
    filters = selected_document_filters(effective_hashes)
    plan = [
        f"Selected tool: {AGENT_TOOL_LABELS.get(tool_name, tool_name)}",
        f"Search mode: {search_mode.title()}",
        f"Evidence target: top {top_k} chunks",
    ]
    if selected_hashes:
        plan.append(f"Manual focus documents: {describe_documents(selected_hashes, documents_by_hash)}")
    if inferred_hashes:
        plan.append(f"Inferred focus documents: {describe_documents(inferred_hashes, documents_by_hash)}")
    if "compare" in goal.lower() and len(effective_hashes) < 2:
        plan.append("Could not identify two focus documents, so comparison may be limited.")

    if tool_name == "search_documents":
        result = generate_rag_result(
            query=goal,
            chat_model=chat_model,
            embedding_model=embedding_model,
            top_k=top_k,
            min_score=min_score,
            response_language_name=response_language_name,
            filters=filters,
            search_mode=search_mode,
        )
        plan.append("Answered directly from retrieved evidence.")
        return {
            "tool": tool_name,
            "plan": plan,
            "answer": result["answer"],
            "result": result,
            "report_markdown": build_agent_report_markdown(goal, tool_name, plan, result),
            "runtime_settings": runtime_settings,
            "search_mode": search_mode,
            "filters": filters,
        }

    if not require_demo_budget(
        "Agentic RAG run",
        estimated_calls=estimated_rag_calls(search_mode),
        active_settings=runtime_settings,
    ):
        raise RAGApplicationError("Demo usage limit reached before agent execution.")

    if tool_name in {"summarize_documents", "compare_documents", "generate_report"} and effective_hashes:
        chunks = collect_document_context_chunks(store, effective_hashes, limit_per_document=8)
        plan.append("Used focused document chunks instead of open-ended search.")
    else:
        retrieval_query = f"{tool_name.replace('_', ' ')}: {goal}"
        chunks = pipeline.retrieve_chunks(
            retrieval_query,
            top_k=top_k,
            min_score=min_score,
            filters=filters,
            search_mode=search_mode,
        )
        plan.append("Retrieved evidence using the user goal.")

    if not chunks:
        result = {
            "answer": "I don't know",
            "sources": [],
            "source_metadata": [],
            "confidence": 0.0,
        }
        return {
            "tool": tool_name,
            "plan": plan,
            "answer": result["answer"],
            "result": result,
            "report_markdown": build_agent_report_markdown(goal, tool_name, plan, result),
            "runtime_settings": runtime_settings,
            "search_mode": search_mode,
            "filters": filters,
        }

    agent_prompt = agent_prompt_for_tool(tool_name, goal, selected_names)
    answer_parts: list[str] = []
    for delta in pipeline.stream_answer(
        pipeline.build_prompt(agent_prompt, chunks, response_language=response_language_name)
    ):
        answer_parts.append(delta)
    answer = "".join(answer_parts).strip() or "I don't know"
    result = pipeline.result_from_answer(answer, chunks)
    return {
        "tool": tool_name,
        "plan": plan,
        "answer": result["answer"],
        "result": result,
        "report_markdown": build_agent_report_markdown(goal, tool_name, plan, result),
        "runtime_settings": runtime_settings,
        "search_mode": search_mode,
        "filters": filters,
    }


def build_agent_report_markdown(goal: str, tool_name: str, plan: list[str], result: dict) -> str:
    lines = [
        "# Agentic RAG Report",
        "",
        f"**Goal:** {goal.strip()}",
        f"**Tool used:** {AGENT_TOOL_LABELS.get(tool_name, tool_name)}",
        f"**Confidence:** {float(result.get('confidence', 0.0) or 0.0):.2f}",
        "",
        "## Agent Plan",
    ]
    lines.extend(f"- {step}" for step in plan)
    lines.extend(["", "## Answer", "", result.get("answer", "I don't know"), "", "## Citations"])
    sources = result.get("sources", [])
    metadata_items = result.get("source_metadata", [])
    if not sources:
        lines.append("- No citations available.")
    for index, (source, metadata) in enumerate(zip(sources, metadata_items), start=1):
        file_name = metadata.get("file_name") or "Unknown"
        page = metadata.get("page_number")
        page_text = f", page {page}" if page else ""
        score = float(metadata.get("score", 0.0) or 0.0)
        excerpt = " ".join(str(source).split())[:320]
        lines.append(f"- Source {index}: {file_name}{page_text}, score {score:.2f}. {excerpt}")
    return "\n".join(lines)


def agent_pdf_report(markdown_text: str) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, title="Agentic RAG Report")
    styles = getSampleStyleSheet()
    story = []
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 8))
            continue
        if line.startswith("# "):
            story.append(Paragraph(html.escape(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(html.escape(line[3:]), styles["Heading2"]))
        elif line.startswith("- "):
            story.append(Paragraph(html.escape(line), styles["BodyText"]))
        else:
            story.append(Paragraph(html.escape(line), styles["BodyText"]))
    document.build(story)
    return buffer.getvalue()


def agent_evidence_dataframe(result: dict) -> pd.DataFrame:
    rows = []
    for metadata in result.get("source_metadata", []):
        rows.append(
            {
                "Document": metadata.get("file_name") or "Unknown",
                "Retrieval": metadata.get("retrieval_method") or "semantic",
                "Score": float(metadata.get("score", 0.0) or 0.0),
                "Page": metadata.get("page_number") or "",
            }
        )
    return pd.DataFrame(rows)


def generate_rag_result(
    *,
    query: str,
    retrieval_query: str | None = None,
    chat_model: str,
    embedding_model: str,
    top_k: int,
    min_score: float,
    response_language_name: str,
    filters: dict | None,
    search_mode: str,
    pinned_chunks: list[dict] | None = None,
    stream_placeholder=None,
) -> dict:
    pipeline = get_pipeline(chat_model, embedding_model)
    if not require_demo_budget(
        "RAG answer generation",
        estimated_calls=estimated_rag_calls(search_mode),
        active_settings=pipeline.settings,
    ):
        raise RAGApplicationError("Demo usage limit reached before answer generation.")
    effective_retrieval_query = retrieval_query or query
    chunks = pipeline.retrieve_chunks(
        effective_retrieval_query,
        top_k=top_k,
        min_score=min_score,
        filters=filters,
        search_mode=search_mode,
    )
    if pinned_chunks:
        chunks = merge_context_chunks(
            pinned_chunks,
            chunks,
            max_chunks=min(20, max(top_k + 4, top_k + len(pinned_chunks))),
        )

    if not chunks:
        result = pipeline.answer_question(
            effective_retrieval_query,
            top_k=top_k,
            min_score=min_score,
            response_language=response_language_name,
            filters=filters,
            search_mode=search_mode,
        )
        if stream_placeholder is not None:
            stream_placeholder.markdown(result["answer"])
        return result

    prompt = pipeline.build_prompt(query, chunks, response_language=response_language_name)
    answer_parts: list[str] = []
    for delta in pipeline.stream_answer(prompt):
        answer_parts.append(delta)
        if stream_placeholder is not None:
            stream_placeholder.markdown("".join(answer_parts) + " |")

    answer = "".join(answer_parts).strip()
    if stream_placeholder is not None:
        stream_placeholder.markdown(answer or "I don't know")
    return pipeline.result_from_answer(answer, chunks)


def merge_context_chunks(primary_chunks: list[dict], secondary_chunks: list[dict], *, max_chunks: int) -> list[dict]:
    merged: list[dict] = []
    seen: set[str] = set()

    for chunk in primary_chunks + secondary_chunks:
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


def default_evaluation_cases() -> list[dict]:
    return [dict(case) for case in DEFAULT_EVALUATION_CASES]


def evaluation_cases_path(active_settings=settings) -> Path:
    return active_settings.data_dir / "evaluation_cases.json"


def normalize_evaluation_cases(rows: list[dict]) -> list[dict]:
    cases: list[dict] = []
    for row in rows:
        question = str(row.get("Question", "") or "").strip()
        if not question:
            continue
        cases.append(
            {
                "Question": question,
                "Expected answer": str(row.get("Expected answer", "") or "").strip(),
                "Expected unknown": bool(row.get("Expected unknown", False)),
                "Required citation contains": str(row.get("Required citation contains", "") or "").strip(),
            }
        )
    return cases


def load_evaluation_cases(active_settings=settings) -> list[dict]:
    path = evaluation_cases_path(active_settings)
    if not path.exists():
        return default_evaluation_cases()
    try:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_evaluation_cases()
    if not isinstance(raw_cases, list):
        return default_evaluation_cases()
    cases = normalize_evaluation_cases([case for case in raw_cases if isinstance(case, dict)])
    return cases or default_evaluation_cases()


def save_evaluation_cases(cases: list[dict], active_settings=settings) -> None:
    active_settings.ensure_directories()
    evaluation_cases_path(active_settings).write_text(
        json.dumps(cases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


EVALUATION_STOP_WORDS = {
    "the",
    "and",
    "that",
    "this",
    "with",
    "from",
    "into",
    "were",
    "was",
    "for",
    "who",
    "what",
    "where",
    "when",
    "how",
    "did",
    "does",
    "are",
    "not",
    "know",
}


def evaluation_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in EVALUATION_STOP_WORDS
    }


def is_unknown_answer(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.lower()).strip()
    return any(
        phrase in normalized
        for phrase in (
            "i don't know",
            "i do not know",
            "not enough information",
            "not provided in the context",
            "cannot answer from the context",
            "answer is not in the context",
        )
    )


def answer_quality_score(answer: str, expected: str, expected_unknown: bool) -> float:
    if expected_unknown:
        return 1.0 if is_unknown_answer(answer) else 0.0

    if is_unknown_answer(answer):
        return 0.0

    expected_terms = evaluation_tokens(expected)
    if not expected_terms:
        return 0.0
    answer_terms = evaluation_tokens(answer)
    return round(len(expected_terms & answer_terms) / len(expected_terms), 3)


def citation_correctness(result: dict, required_text: str) -> bool | None:
    required = required_text.strip().lower()
    if not required:
        return None
    for metadata in result.get("source_metadata", []):
        file_name = str(metadata.get("file_name", "")).lower()
        source_path = str(metadata.get("source_path", "")).lower()
        if required in file_name or required in source_path:
            return True
    return False


def evaluate_result_row(case: dict, result: dict, *, search_mode: str) -> dict:
    answer = str(result.get("answer", "") or "")
    expected_unknown = bool(case.get("Expected unknown"))
    citation_match = citation_correctness(result, str(case.get("Required citation contains", "") or ""))
    citation_score = None if citation_match is None else (1.0 if citation_match else 0.0)
    idk_correct = is_unknown_answer(answer) if expected_unknown else not is_unknown_answer(answer)
    quality = answer_quality_score(answer, str(case.get("Expected answer", "") or ""), expected_unknown)
    overall_scores = [quality, 1.0 if idk_correct else 0.0]
    if citation_score is not None:
        overall_scores.append(citation_score)
    sources = result.get("source_metadata", [])
    top_source = sources[0].get("file_name", "") if sources else ""
    overall = sum(overall_scores) / len(overall_scores)
    return {
        "Question": case["Question"],
        "Expected": case.get("Expected answer", ""),
        "Answer": answer,
        "Retrieval score": round(float(result.get("confidence", 0.0) or 0.0), 3),
        "Answer quality": quality,
        "Citation correctness": "N/A" if citation_match is None else ("Pass" if citation_match else "Fail"),
        "I don't know accuracy": "Pass" if idk_correct else "Fail",
        "Overall": round(overall, 3),
        "Sources": len(result.get("sources", [])),
        "Top source": top_source,
        "Search mode": search_mode,
        "Status": "Pass" if overall >= 0.7 else "Review",
        "Error": "",
    }


def evaluation_error_row(case: dict, exc: BaseException, *, search_mode: str) -> dict:
    return {
        "Question": case.get("Question", ""),
        "Expected": case.get("Expected answer", ""),
        "Answer": "",
        "Retrieval score": 0.0,
        "Answer quality": 0.0,
        "Citation correctness": "N/A",
        "I don't know accuracy": "Fail",
        "Overall": 0.0,
        "Sources": 0,
        "Top source": "",
        "Search mode": search_mode,
        "Status": "Error",
        "Error": compact_error_detail(exc),
    }


def top_source_documents(result: dict, limit: int = 3) -> list[str]:
    counts: Counter[str] = Counter()
    for metadata in result.get("source_metadata", []):
        file_name = metadata.get("file_name")
        if file_name:
            counts[str(file_name)] += 1
    return [name for name, _ in counts.most_common(limit)]


def persist_uploaded_file(uploaded_file, runtime_settings) -> tuple[Path, str, str]:
    original_name = safe_filename(uploaded_file.name)
    extension = Path(original_name).suffix.lower()

    if extension not in runtime_settings.allowed_extensions:
        allowed = ", ".join(runtime_settings.allowed_extensions)
        raise RAGApplicationError(f"Unsupported file type '{extension}'. Allowed: {allowed}.")

    runtime_settings.ensure_directories()
    temp_path = runtime_settings.upload_dir / f"pending_{uuid4().hex}_{original_name}"
    max_upload_mb = demo_upload_size_limit_mb(runtime_settings)
    max_bytes = max_upload_mb * 1024 * 1024
    digest = hashlib.sha256()
    size = 0

    uploaded_file.seek(0)
    with temp_path.open("wb") as output:
        while True:
            block = uploaded_file.read(1024 * 1024)
            if not block:
                break

            size += len(block)
            if size > max_bytes:
                output.close()
                temp_path.unlink(missing_ok=True)
                raise RAGApplicationError(
                    f"{original_name} exceeds the {format_size(max_upload_mb)} per-file demo limit."
                )

            digest.update(block)
            output.write(block)

    if size == 0:
        temp_path.unlink(missing_ok=True)
        raise RAGApplicationError(f"{original_name} is empty.")

    file_hash = digest.hexdigest()
    saved_path = runtime_settings.upload_dir / f"{file_hash[:12]}_{original_name}"
    temp_path.replace(saved_path)
    uploaded_file.seek(0)
    return saved_path, file_hash, original_name


def save_and_ingest(
    uploaded_file,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    runtime_settings,
) -> dict:
    saved_path, file_hash, original_name = persist_uploaded_file(uploaded_file, runtime_settings)
    return ingest_saved_path(
        saved_path,
        file_hash,
        original_name,
        chunk_size_tokens,
        chunk_overlap_tokens,
        runtime_settings,
    )


def ingest_saved_path(
    saved_path: Path,
    file_hash: str,
    original_name: str,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    runtime_settings,
) -> dict:
    store = get_vector_store(runtime_settings.openai_embedding_model)
    existing_document = store.get_document(file_hash)
    if existing_document:
        logger.info("Duplicate upload skipped for %s", original_name)
        Path(saved_path).unlink(missing_ok=True)
        return {
            "file_name": existing_document.get("file_name", original_name),
            "file_hash": file_hash,
            "chunks_added": 0,
            "total_chunks": existing_document.get("chunk_count", 0),
            "skipped": True,
        }

    with get_store_lock():
        return ingest_file(
            Path(saved_path),
            store,
            file_hash=file_hash,
            display_name=original_name,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            active_settings=runtime_settings,
        )


def enqueue_uploaded_files(
    uploaded_files,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
    runtime_settings,
) -> tuple[int, list[str]]:
    added = 0
    failures: list[str] = []
    max_files = demo_upload_file_limit(runtime_settings)
    for uploaded_file in uploaded_files or []:
        if added >= max_files:
            failures.append(f"{uploaded_file.name}: demo upload batch limit is {max_files} file(s).")
            continue
        try:
            saved_path, file_hash, original_name = persist_uploaded_file(uploaded_file, runtime_settings)
            st.session_state.ingestion_queue.append(
                {
                    "id": uuid4().hex,
                    "file_name": original_name,
                    "file_hash": file_hash,
                    "saved_path": str(saved_path),
                    "chunk_size_tokens": chunk_size_tokens,
                    "chunk_overlap_tokens": chunk_overlap_tokens,
                    "embedding_model": runtime_settings.openai_embedding_model,
                    "vision_model": runtime_settings.openai_vision_model,
                    "vision_enabled": runtime_settings.vision_ingestion_enabled,
                    "vision_detail": runtime_settings.vision_detail,
                    "status": "queued",
                    "attempts": 0,
                    "chunks_added": 0,
                    "message": "Queued",
                    "queued_at": datetime.now().strftime("%H:%M:%S"),
                }
            )
            added += 1
        except Exception as exc:
            failures.append(f"{uploaded_file.name}: {exc}")
    return added, failures


def process_ingestion_queue(runtime_settings) -> dict:
    queued_items = [item for item in st.session_state.ingestion_queue if item.get("status") == "queued"]
    summary = {"processed": 0, "chunks_added": 0, "skipped": 0, "failed": 0}
    if not queued_items:
        return summary

    progress = st.progress(0)
    status = st.empty()
    for index, item in enumerate(queued_items, start=1):
        item["status"] = "indexing"
        item["attempts"] = int(item.get("attempts", 0)) + 1
        item["message"] = "Extracting, chunking, embedding, and indexing"
        status.info(
            f"Processing {index} of {len(queued_items)}: {item['file_name']} "
            "(extract > chunk > embed > index)"
        )
        try:
            item_settings = active_settings(
                embedding_model=item.get("embedding_model", runtime_settings.openai_embedding_model),
                vision_model=item.get("vision_model", runtime_settings.openai_vision_model),
                vision_ingestion_enabled=bool(item.get("vision_enabled", runtime_settings.vision_ingestion_enabled)),
                vision_detail=item.get("vision_detail", runtime_settings.vision_detail),
            )
            if not require_demo_budget("Document ingestion", estimated_calls=2, active_settings=item_settings):
                item["status"] = "failed"
                item["message"] = "Paused by public demo usage limit"
                item["failed_at"] = datetime.now().strftime("%H:%M:%S")
                summary["failed"] += 1
                progress.progress(index / len(queued_items))
                continue
            result = ingest_saved_path(
                Path(item["saved_path"]),
                item["file_hash"],
                item["file_name"],
                int(item["chunk_size_tokens"]),
                int(item["chunk_overlap_tokens"]),
                item_settings,
            )
            item["status"] = "completed"
            item["chunks_added"] = result.get("chunks_added", 0)
            item["message"] = "Duplicate skipped" if result.get("skipped") else "Indexed in FAISS"
            item["completed_at"] = datetime.now().strftime("%H:%M:%S")
            summary["processed"] += 1
            summary["chunks_added"] += int(result.get("chunks_added", 0))
            summary["skipped"] += 1 if result.get("skipped") else 0
        except Exception as exc:
            item["status"] = "failed"
            item["message"] = str(exc) or exc.__class__.__name__
            item["failed_at"] = datetime.now().strftime("%H:%M:%S")
            summary["failed"] += 1
        progress.progress(index / len(queued_items))

    clear_index_caches()
    status.success(
        f"Queue completed. {summary['processed']} processed, "
        f"{summary['chunks_added']} chunks added, {summary['failed']} failed."
    )
    return summary


def queue_rows() -> list[dict]:
    return [
        {
            "File": item.get("file_name"),
            "Status": item.get("status"),
            "Chunks": item.get("chunks_added", 0),
            "Attempts": item.get("attempts", 0),
            "Message": item.get("message", ""),
        }
        for item in st.session_state.ingestion_queue
    ]


def clear_index_caches() -> None:
    get_pipeline.clear()
    get_vector_store.clear()


def reindex_document(document: dict, runtime_settings, *, force: bool = True) -> dict:
    file_hash = document["file_hash"]
    source_path = Path(document["source_path"]) if document.get("source_path") else None
    if not source_path or not source_path.exists():
        source_path = resolve_document_path(file_hash, runtime_settings)
    if not source_path:
        raise RAGApplicationError(f"Stored file is missing for {document.get('file_name', file_hash)}.")

    store = get_vector_store(runtime_settings.openai_embedding_model)
    if force:
        store.remove_document(file_hash)

    return ingest_file(
        source_path,
        store,
        file_hash=file_hash,
        display_name=document.get("file_name") or source_path.name,
        active_settings=runtime_settings,
    )


def migrate_document(document: dict, source_settings, target_settings, *, force: bool = False) -> dict:
    file_hash = document["file_hash"]
    source_path = Path(document["source_path"]) if document.get("source_path") else None
    if not source_path or not source_path.exists():
        source_path = resolve_document_path(file_hash, source_settings)
    if not source_path:
        raise RAGApplicationError(f"Stored file is missing for {document.get('file_name', file_hash)}.")

    target_store = get_vector_store(target_settings.openai_embedding_model)
    if force:
        target_store.remove_document(file_hash)

    return ingest_file(
        source_path,
        target_store,
        file_hash=file_hash,
        display_name=document.get("file_name") or source_path.name,
        active_settings=target_settings,
    )


def render_metric_grid(embedding_model: str | None = None) -> None:
    stats = index_stats(embedding_model)
    history_count = len(st.session_state.query_history)
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Documents</div>
                <div class="metric-value">{stats["documents"]}</div>
                <div class="metric-note">Current embedding index</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Indexed Chunks</div>
                <div class="metric-value">{stats["chunks"]}</div>
                <div class="metric-note">FAISS cosine vectors</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Queries This Session</div>
                <div class="metric-value">{history_count}</div>
                <div class="metric-note">Local browser session</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Last Indexed</div>
                <div class="metric-value" style="font-size: 1.05rem;">{stats["latest"]}</div>
                <div class="metric-note">{stats["model"]}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">Enterprise RAG Console</div>
            <div class="sidebar-brand-subtitle">Knowledge retrieval workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_identity_controls()
    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-label">Appearance</div>', unsafe_allow_html=True)
    theme_mode = st.sidebar.segmented_control(
        "Theme",
        ["Dark", "Light"],
        default=st.session_state.theme_mode,
        label_visibility="collapsed",
    )
    if theme_mode != st.session_state.theme_mode:
        st.session_state.theme_mode = theme_mode
        st.rerun()
    st.sidebar.divider()
    for label, group_items in NAV_GROUPS:
        st.sidebar.markdown(f'<div class="sidebar-section-label">{escape_html(label)}</div>', unsafe_allow_html=True)
        for item in group_items:
            disabled = not can_access_nav(item)
            button_type = "primary" if st.session_state.nav_selection == item else "secondary"
            if st.sidebar.button(
                item,
                key=f"nav_{item}",
                width="stretch",
                type=button_type,
                disabled=disabled,
            ):
                st.session_state.nav_selection = item
                st.rerun()

    if not can_access_nav(st.session_state.nav_selection):
        st.session_state.nav_selection = default_nav_selection()

    selected = st.session_state.nav_selection
    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section-label">Runtime</div>', unsafe_allow_html=True)
    st.sidebar.write(f"Embedding index: `{active_embedding_model()}`")
    st.sidebar.write(f"Chat model: `{active_chat_model()}`")
    st.sidebar.write(f"Vision model: `{active_vision_model()}`")
    st.sidebar.write(f"SSL: `{ssl_runtime_description(settings)}`")
    render_demo_limit_status(active_settings())
    st.sidebar.divider()

    if st.sidebar.button("Refresh local index", width="stretch"):
        get_pipeline.clear()
        get_vector_store.clear()
        st.rerun()

    if st.sidebar.button("Test OpenAI connection", width="stretch"):
        try:
            runtime_settings = active_settings()
            if require_demo_budget("OpenAI connection test", active_settings=runtime_settings):
                with st.sidebar.status("Calling embeddings API", expanded=False):
                    generate_embeddings(["connection test"], active_settings=runtime_settings)
                st.sidebar.success("Connection verified")
        except RAGApplicationError as exc:
            st.sidebar.error(exc.message)
        except Exception as exc:
            logger.exception("OpenAI connection test failed: %s", exc)
            st.sidebar.error(str(exc) or exc.__class__.__name__)

    return selected


def render_dashboard() -> None:
    render_header("Knowledge Operations", "Monitor index readiness, ingestion volume, and retrieval activity.")
    render_metric_grid()

    action_col_a, action_col_b, action_col_c = st.columns([1, 1, 1], gap="small")
    if action_col_a.button("Ask a question", key="dashboard_go_ask", type="primary", width="stretch"):
        st.session_state.nav_selection = "Ask"
        st.rerun()
    if action_col_b.button("Open conversation", key="dashboard_go_conversation", width="stretch"):
        st.session_state.nav_selection = "Conversation"
        st.rerun()
    if is_admin() and action_col_c.button("Upload documents", key="dashboard_go_ingestion", width="stretch"):
        st.session_state.nav_selection = "Ingestion"
        st.rerun()

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader("Indexed Documents")
        rows = document_rows()
        if rows:
            st.dataframe(rows[:8], hide_index=True, width="stretch")
        else:
            st.info("No documents indexed for the current embedding model.")

    with right:
        st.subheader("Governance")
        st.markdown(
            """
            <div class="section-panel">
                <span class="small-pill">Source-grounded answers</span>
                <span class="small-pill">Duplicate detection</span>
                <span class="small-pill">Local FAISS cache</span>
                <span class="small-pill">Source audit trail</span>
                <span class="small-pill">Folder ingestion</span>
                <span class="small-pill">Corporate SSL mode</span>
                <span class="small-pill">Role-based access</span>
                <span class="small-pill">Index lifecycle controls</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.subheader("Session Activity")
        if st.session_state.query_history:
            for item in reversed(st.session_state.query_history[-5:]):
                st.write(f"{item['time']} - {item['query'][:80]}")
        else:
            st.caption("No questions asked in this session.")


def render_ingestion() -> None:
    render_header("Ingestion Center", "Add files or folders to the local knowledge base.")
    if not require_admin_ui():
        return

    left, right = st.columns([1.5, 1], gap="large")
    with left:
        upload_mode = st.segmented_control("Upload type", ["Files", "Folder"], default="Files")
        accept_mode = "directory" if upload_mode == "Folder" else True
        uploaded_files = st.file_uploader(
            "Documents",
            type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "webp", "gif"],
            accept_multiple_files=accept_mode,
            key=f"documents_{upload_mode.lower()}",
        )
        selected_count = len(uploaded_files or [])
        st.caption(
            f"{selected_count} selected. Demo batch limit: {demo_upload_file_limit(settings)}. "
            f"Per-file demo limit: {format_size(demo_upload_size_limit_mb(settings))}."
        )
        render_demo_limit_status(settings)

    embedding_model = active_embedding_model()
    vision_enabled = active_vision_enabled()
    vision_model = active_vision_model()
    vision_detail = active_vision_detail()
    runtime_settings = active_settings(
        embedding_model=embedding_model,
        vision_model=vision_model,
        vision_ingestion_enabled=vision_enabled,
        vision_detail=vision_detail,
    )
    chunk_size = runtime_settings.chunk_size_tokens
    chunk_overlap = runtime_settings.chunk_overlap_tokens

    with right:
        st.subheader("Indexing Defaults")
        st.caption(f"Index: {runtime_settings.index_dir.name}")
        st.caption(f"Chunking: {chunk_size} tokens with {chunk_overlap} overlap")
        with st.expander("Advanced indexing settings", expanded=False):
            st.subheader("Index Model")
            embedding_model = model_selectbox(
                "Embedding model",
                EMBEDDING_MODEL_OPTIONS,
                active_embedding_model(),
                "ingestion_embedding_model",
            )
            st.session_state.embedding_model = embedding_model

            st.subheader("Visual Understanding")
            vision_enabled = st.toggle(
                "Understand images during indexing",
                value=active_vision_enabled(),
                help="Uses the vision model to describe images in PDFs, DOCX files, and standalone image uploads.",
            )
            vision_model = model_selectbox(
                "Vision model",
                VISION_MODEL_OPTIONS,
                active_vision_model(),
                "ingestion_vision_model",
                disabled=not vision_enabled,
            )
            vision_detail = st.selectbox(
                "Vision detail",
                ["high", "auto", "low"],
                index=["high", "auto", "low"].index(
                    active_vision_detail() if active_vision_detail() in {"high", "auto", "low"} else "high"
                ),
                disabled=not vision_enabled,
            )
            st.session_state.vision_ingestion_enabled = vision_enabled
            st.session_state.vision_model = vision_model
            st.session_state.vision_detail = vision_detail

            runtime_settings = active_settings(
                embedding_model=embedding_model,
                vision_model=vision_model,
                vision_ingestion_enabled=vision_enabled,
                vision_detail=vision_detail,
            )

            st.subheader("Chunking")
            chunk_size = st.number_input(
                "Chunk size",
                min_value=100,
                max_value=4000,
                value=runtime_settings.chunk_size_tokens,
                step=50,
            )
            chunk_overlap = st.number_input(
                "Overlap",
                min_value=0,
                max_value=max(0, int(chunk_size) - 1),
                value=min(runtime_settings.chunk_overlap_tokens, max(0, int(chunk_size) - 1)),
                step=10,
            )

    runtime_settings = active_settings(
        embedding_model=st.session_state.embedding_model,
        vision_model=st.session_state.vision_model,
        vision_ingestion_enabled=st.session_state.vision_ingestion_enabled,
        vision_detail=st.session_state.vision_detail,
    )
    upload_file_limit = demo_upload_file_limit(runtime_settings)
    start_disabled = not uploaded_files or len(uploaded_files) > upload_file_limit
    if uploaded_files and len(uploaded_files) > upload_file_limit:
        st.error(f"Select {upload_file_limit} documents or fewer per demo batch.")

    upload_action_col, _ = st.columns([1, 2])
    if upload_action_col.button("Add to queue", type="primary", disabled=start_disabled, width="stretch"):
        added_to_queue, failures = enqueue_uploaded_files(
            uploaded_files,
            int(chunk_size),
            int(chunk_overlap),
            runtime_settings,
        )
        if failures:
            st.warning(f"Queued {added_to_queue}; {len(failures)} failed before queueing.")
            with st.expander("Queue failures", expanded=True):
                for failure in failures:
                    st.write(failure)
        else:
            st.success(f"Queued {added_to_queue} document(s).")

    queued_count = sum(1 for item in st.session_state.ingestion_queue if item.get("status") == "queued")
    failed_count = sum(1 for item in st.session_state.ingestion_queue if item.get("status") == "failed")
    indexing_count = sum(1 for item in st.session_state.ingestion_queue if item.get("status") == "indexing")
    if indexing_count:
        active_ingestion_step = "index"
    elif queued_count:
        active_ingestion_step = "extract"
    elif st.session_state.last_ingestion:
        active_ingestion_step = "index"
    else:
        active_ingestion_step = "upload"
    render_ingestion_steps(active_ingestion_step)

    queue_col_b, queue_col_c = st.columns([1, 1])
    if queue_col_b.button("Start queue", disabled=queued_count == 0, width="stretch"):
        summary = process_ingestion_queue(runtime_settings)
        summary = {
            "time": datetime.now().strftime("%H:%M:%S"),
            **summary,
        }
        st.session_state.last_ingestion = summary

    if queue_col_c.button("Retry failed", disabled=failed_count == 0, width="stretch"):
        for item in st.session_state.ingestion_queue:
            if item.get("status") == "failed":
                item["status"] = "queued"
                item["message"] = "Queued for retry"
        st.rerun()

    if st.session_state.ingestion_queue:
        st.subheader("Processing Queue")
        st.dataframe(queue_rows(), hide_index=True, width="stretch")
        clear_col_a, clear_col_b = st.columns([1, 3])
        if clear_col_a.button("Clear completed", width="stretch"):
            st.session_state.ingestion_queue = [
                item for item in st.session_state.ingestion_queue if item.get("status") != "completed"
            ]
            st.rerun()

    if st.session_state.last_ingestion:
        st.subheader("Latest Ingestion")
        item = st.session_state.last_ingestion
        st.write(
            f"{item['time']} - {item['processed']} processed, {item['chunks_added']} chunks added, "
            f"{item['skipped']} duplicates skipped, {item['failed']} failed."
        )


def render_answer_sources(
    result: dict,
    min_score: float,
    runtime_settings,
    *,
    key_prefix: str = "sources",
) -> None:
    source_count = len(result["sources"])
    label, tone, _ = confidence_status(result)
    st.markdown(
        f"""
        <div class="evidence-summary">
            <div class="evidence-chip">
                <div class="evidence-chip-label">Confidence</div>
                <div class="evidence-chip-value"><span class="confidence-badge confidence-{tone}">{escape_html(label)}</span></div>
            </div>
            <div class="evidence-chip">
                <div class="evidence-chip-label">Sources</div>
                <div class="evidence-chip-value">{source_count}</div>
            </div>
            <div class="evidence-chip">
                <div class="evidence-chip-label">Minimum similarity</div>
                <div class="evidence-chip-value">{min_score:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not result["sources"]:
        st.info("No retrieved context met the selected threshold.")
        return

    st.subheader("Source Evidence")
    for index, (source, metadata) in enumerate(zip(result["sources"], result["source_metadata"]), start=1):
        file_name = metadata.get("file_name") or "Unknown"
        page = metadata.get("page_number")
        page_label = f"Page {page}" if page else "Page not available"
        source_type = metadata.get("source_type") or "text"
        retrieval_method = metadata.get("retrieval_method") or "semantic"
        image_index = metadata.get("image_index")
        image_label = f"Image {image_index}" if image_index else ""
        score = metadata.get("score", 0)
        score_value = float(score or 0.0)
        card_meta_parts = [page_label, source_type, retrieval_method]
        if image_label:
            card_meta_parts.append(image_label)
        card_meta = " - ".join(str(part) for part in card_meta_parts if part)
        score_tone = "high" if score_value >= 0.72 else ("medium" if score_value >= 0.45 else "low")
        excerpt = source[:850].strip()
        if len(source) > 850:
            excerpt = f"{excerpt}..."
        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-card-header">
                    <div>
                        <div class="source-card-title">Source {index}: {escape_html(file_name)}</div>
                        <div class="source-card-meta">{escape_html(card_meta)}</div>
                    </div>
                    <span class="confidence-badge confidence-{score_tone}">Score {score_value:.2f}</span>
                </div>
                <div class="source-card-excerpt">{escape_html(excerpt)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(
            f"Inspect source {index}",
            expanded=False,
        ):
            st.markdown(
                f"""
                <div class="source-panel">
                    <div class="source-meta">Source {index} - {escape_html(file_name)} - {escape_html(card_meta)} - score {score_value:.2f}</div>
                    <div>{escape_html(source[:1200])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            meta_col, action_col = st.columns([1.2, 0.8], gap="large")
            with meta_col:
                st.dataframe(source_metadata_rows(metadata), hide_index=True, width="stretch")
                st.code(source, language="text")
            with action_col:
                with st.expander("Preview source", expanded=index == 1):
                    render_source_preview(metadata, runtime_settings)

                st.download_button(
                    "Download citation text",
                    source,
                    citation_filename(metadata),
                    "text/plain",
                    key=f"{key_prefix}_source_download_{index}",
                    width="stretch",
                )

                source_path = resolve_document_path(metadata.get("file_hash"), runtime_settings)
                if source_path and source_path.exists():
                    st.link_button(
                        "Open stored document",
                        source_path.resolve().as_uri(),
                        width="stretch",
                    )
                    st.download_button(
                        "Download stored document",
                        source_path.read_bytes(),
                        source_path.name,
                        key=f"{key_prefix}_document_download_{index}",
                        width="stretch",
                    )
                else:
                    st.caption("Original stored file is not available.")


def render_ask() -> None:
    render_header("Ask Workspace", "Generate source-grounded answers with auditable retrieval context.")
    with st.container(key="ask_chat_shell"):
        with st.container(border=True, key="ask_settings"):
            with st.expander("Advanced controls", expanded=False):
                model_col, retrieval_col, voice_col = st.columns([1, 1.15, 1], gap="large")
                with model_col:
                    st.subheader("Models")
                    chat_model = model_selectbox(
                        "Answer model",
                        CHAT_MODEL_OPTIONS,
                        active_chat_model(),
                        "ask_chat_model",
                        disabled=not can_change_models(),
                    )
                    embedding_model = model_selectbox(
                        "Knowledge index",
                        EMBEDDING_MODEL_OPTIONS,
                        active_embedding_model(),
                        "ask_embedding_model",
                        disabled=not can_change_models(),
                    )
                    if not can_change_models():
                        chat_model = active_chat_model()
                        embedding_model = active_embedding_model()
                        st.caption("Model changes require Admin role.")

                st.session_state.chat_model = chat_model
                st.session_state.embedding_model = embedding_model
                store = get_vector_store(embedding_model)

                with retrieval_col:
                    st.subheader("Retrieval")
                    search_mode = st.segmented_control(
                        "Search mode",
                        ["Hybrid", "Semantic", "Keyword"],
                        default="Hybrid",
                        key="ask_search_mode",
                        help="Hybrid combines semantic FAISS search with keyword/BM25-style matching.",
                    )
                    base_runtime_settings = active_settings(chat_model=chat_model, embedding_model=embedding_model)
                    normalize_demo_top_k_state("ask_top_k", base_runtime_settings)
                    top_k = st.slider(
                        "Top K",
                        min_value=1,
                        max_value=demo_top_k_limit(base_runtime_settings),
                        value=demo_top_k_value(base_runtime_settings),
                        key="ask_top_k",
                    )
                    min_score = st.slider(
                        "Minimum similarity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        key="ask_min_score",
                    )
                    show_prompt_policy = st.toggle("Show prompt policy", value=False, key="ask_show_prompt_policy")
                    if show_prompt_policy:
                        st.code(
                            "Answer ONLY from the provided context. Always answer in the user language.",
                            language="text",
                        )
                    source_filters = render_source_filters("ask", store, use_expander=False)

                with voice_col:
                    voice_settings = render_voice_settings("ask", compact=True)

        runtime_settings = active_settings(
            chat_model=chat_model,
            embedding_model=embedding_model,
            transcription_model=voice_settings["transcription_model"],
            tts_model=voice_settings["tts_model"],
            tts_voice=voice_settings["tts_voice"],
        )

        action_left, action_new = st.columns([1, 0.2], gap="small")
        with action_left:
            st.markdown(
                f"""
                <div class="conversation-action-row">
                    <div>
                        <div class="conversation-action-title">Enterprise RAG Ask</div>
                        <div class="conversation-action-meta">{escape_html(active_embedding_model())} - latest answer workspace</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action_new:
            if st.button("New chat", key="ask_new_chat", width="stretch"):
                reset_ask_chat()
                st.rerun()

        if store.is_empty:
            render_index_empty_state("Ask", "ask_empty_index")
            return

        render_voice_input(
            "ask",
            "Speak your question",
            voice_settings,
            runtime_settings,
            target_text_key="ask_voice_review",
        )
        voice_prompt = ""
        send_voice_prompt = False
        if st.session_state.get("ask_voice_review"):
            voice_prompt = st.text_area(
                "Review voice question",
                key="ask_voice_review",
                height=90,
            )
            send_voice_prompt = st.button(
                "Send voice question",
                type="primary",
                disabled=not voice_prompt.strip(),
                width="stretch",
            )

        if not st.session_state.last_ask_result:
            st.markdown(
                """
                <div class="conversation-empty-state">
                    <div>
                        <div class="conversation-empty-title">What would you like to ask?</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.session_state.last_ask_result:
            last = st.session_state.last_ask_result
            result = last["result"]
            with st.chat_message("user"):
                st.write(last["query"])
            with st.chat_message("assistant"):
                st.write(result["answer"])
                render_answer_quality(result, last["min_score"])
                render_spoken_answer(
                    result["answer"],
                    last["voice_settings"],
                    last["runtime_settings"],
                    language=last["answer_language"],
                    key_prefix="ask_answer",
                )
                with st.expander("Feedback", expanded=False):
                    render_feedback_controls(
                        feedback_key=f"ask_{last['feedback_key']}",
                        query=last["query"],
                        answer=result["answer"],
                        result=result,
                        runtime_settings=last["runtime_settings"],
                        search_mode=last["search_mode"],
                        filters=last["filters"],
                        context="ask",
                    )
                with st.expander("Citations", expanded=False):
                    render_answer_sources(result, last["min_score"], last["runtime_settings"], key_prefix="ask")

        prompt = st.chat_input("Ask Enterprise RAG")
        query = voice_prompt.strip() if send_voice_prompt else (prompt or "").strip()
        if query:
            answer_language = response_language(voice_settings["language"], query)
            previous_ask = st.session_state.last_ask_result or {}
            effective_filters = follow_up_filters(
                query,
                source_filters,
                dominant_source_hashes(previous_ask.get("result")),
                store.documents,
            )
            retrieval_query = contextual_follow_up_query(query, previous_ask, store.documents)
            with st.chat_message("user"):
                st.write(query)
            with st.chat_message("assistant"):
                stream_placeholder = st.empty()
                with st.status("Retrieving context and streaming answer", expanded=False):
                    result = generate_rag_result(
                        query=query,
                        retrieval_query=retrieval_query,
                        chat_model=chat_model,
                        embedding_model=embedding_model,
                        top_k=top_k,
                        min_score=min_score,
                        response_language_name=answer_language,
                        filters=effective_filters,
                        search_mode=str(search_mode).lower(),
                        stream_placeholder=stream_placeholder,
                    )
                stream_placeholder.markdown(result["answer"])
                render_answer_quality(result, min_score)
                render_spoken_answer(
                    result["answer"],
                    voice_settings,
                    runtime_settings,
                    language=answer_language,
                    key_prefix="ask_latest",
                )

            feedback_key = hashlib.sha256(
                "|".join(
                    [
                        query.strip(),
                        result.get("answer", ""),
                        chat_model,
                        embedding_model,
                        datetime.now().isoformat(timespec="seconds"),
                    ]
                ).encode("utf-8")
            ).hexdigest()
            st.session_state.last_ask_result = {
                "feedback_key": feedback_key,
                "query": query.strip(),
                "answer_language": answer_language,
                "result": result,
                "runtime_settings": runtime_settings,
                "voice_settings": voice_settings,
                "min_score": min_score,
                "search_mode": str(search_mode).lower(),
                "filters": effective_filters,
            }

            st.session_state.query_history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "query": query.strip(),
                    "language": answer_language,
                    "model": chat_model,
                    "index": embedding_model,
                    "search_mode": str(search_mode).lower(),
                    "confidence": result["confidence"],
                    "sources": len(result["sources"]),
                    "top_documents": "; ".join(top_source_documents(result)),
                }
            )
            st.rerun()

        if st.session_state.query_history:
            with st.expander("Query history"):
                st.dataframe(
                    list(reversed(st.session_state.query_history)),
                    hide_index=True,
                    width="stretch",
                )


def conversation_context_prompt(new_prompt: str) -> str:
    recent_turns = st.session_state.conversation_messages[-8:]
    if not recent_turns:
        return new_prompt

    lines = [
        "Use the conversation so far only to resolve references in the current question.",
        "Conversation so far:",
    ]
    for message in recent_turns:
        role = "User" if message["role"] == "user" else "Assistant"
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("")
    lines.append(f"Current question to answer: {new_prompt}")
    return "\n".join(lines)


def conversation_export_markdown() -> str:
    lines = ["# Enterprise RAG Conversation", ""]
    for message in st.session_state.conversation_messages:
        role = "User" if message["role"] == "user" else "Assistant"
        lines.append(f"## {role}")
        lines.append(message.get("content", ""))
        citations = message.get("citations") or []
        if citations:
            lines.append("")
            lines.append("### Citations")
            for citation in citations:
                lines.append(
                    f"- {citation.get('file_name')} | chunk {citation.get('chunk_index')} | "
                    f"score {citation.get('score')}"
                )
        lines.append("")
    return "\n".join(lines)


def render_conversation() -> None:
    render_header("Conversation Mode", "Run multi-turn RAG with retained chat context and exportable citations.")
    with st.container(key="conversation_chat_shell"):
        with st.container(border=True, key="conversation_settings"):
            with st.expander("Advanced controls", expanded=False):
                model_col, retrieval_col, voice_col = st.columns([1, 1.15, 1], gap="large")
                with model_col:
                    st.subheader("Models")
                    chat_model = model_selectbox(
                        "Answer model",
                        CHAT_MODEL_OPTIONS,
                        active_chat_model(),
                        "conversation_chat_model",
                        disabled=not can_change_models(),
                    )
                    embedding_model = model_selectbox(
                        "Knowledge index",
                        EMBEDDING_MODEL_OPTIONS,
                        active_embedding_model(),
                        "conversation_embedding_model",
                        disabled=not can_change_models(),
                    )
                    if not can_change_models():
                        chat_model = active_chat_model()
                        embedding_model = active_embedding_model()
                        st.caption("Model changes require Admin role.")

                st.session_state.chat_model = chat_model
                st.session_state.embedding_model = embedding_model
                store = get_vector_store(embedding_model)

                with retrieval_col:
                    st.subheader("Retrieval")
                    search_mode = st.segmented_control(
                        "Search mode",
                        ["Hybrid", "Semantic", "Keyword"],
                        default="Hybrid",
                        key="conversation_search_mode",
                        help="Hybrid combines semantic FAISS search with keyword/BM25-style matching.",
                    )
                    base_runtime_settings = active_settings(chat_model=chat_model, embedding_model=embedding_model)
                    normalize_demo_top_k_state("conversation_top_k", base_runtime_settings)
                    top_k = st.slider(
                        "Top K",
                        min_value=1,
                        max_value=demo_top_k_limit(base_runtime_settings),
                        value=demo_top_k_value(base_runtime_settings),
                        key="conversation_top_k",
                    )
                    min_score = st.slider(
                        "Minimum similarity",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.01,
                        key="conversation_min_score",
                    )
                    source_filters = render_source_filters("conversation", store, use_expander=False)

                with voice_col:
                    voice_settings = render_voice_settings("conversation", compact=True)

        runtime_settings = active_settings(
            chat_model=chat_model,
            embedding_model=embedding_model,
            transcription_model=voice_settings["transcription_model"],
            tts_model=voice_settings["tts_model"],
            tts_voice=voice_settings["tts_voice"],
        )

        action_left, action_new, action_export_md, action_export_json = st.columns([1, 0.2, 0.18, 0.16], gap="small")
        with action_left:
            st.markdown(
                f"""
                <div class="conversation-action-row">
                    <div>
                        <div class="conversation-action-title">Enterprise RAG Chat</div>
                        <div class="conversation-action-meta">{escape_html(active_embedding_model())} - {len(st.session_state.conversation_messages)} messages</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action_new:
            if st.button("New chat", key="conversation_new_chat", width="stretch"):
                reset_conversation_chat()
                st.rerun()
        if st.session_state.conversation_messages:
            with action_export_md:
                st.download_button(
                    "Markdown",
                    conversation_export_markdown(),
                    "rag_conversation.md",
                    "text/markdown",
                    width="stretch",
                )
            with action_export_json:
                st.download_button(
                    "JSON",
                    json.dumps(st.session_state.conversation_messages, indent=2),
                    "rag_conversation.json",
                    "application/json",
                    width="stretch",
                )

        if store.is_empty:
            render_index_empty_state("Conversation", "conversation_empty_index")
            return

        render_voice_input(
            "conversation",
            "Speak a follow-up question",
            voice_settings,
            runtime_settings,
            target_text_key="conversation_voice_review",
        )
        voice_prompt = ""
        send_voice_prompt = False
        if st.session_state.get("conversation_voice_review"):
            voice_prompt = st.text_area(
                "Review voice question",
                key="conversation_voice_review",
                height=90,
            )
            send_voice_prompt = st.button(
                "Send voice question",
                type="primary",
                disabled=not voice_prompt.strip(),
                width="stretch",
            )

        if not st.session_state.conversation_messages:
            st.markdown(
                """
                <div class="conversation-empty-state">
                    <div>
                        <div class="conversation-empty-title">What would you like to know?</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for message_index, message in enumerate(st.session_state.conversation_messages):
            with st.chat_message(message["role"]):
                st.write(message.get("content", ""))
                if (
                    message["role"] == "assistant"
                    and message.get("spoken_answer")
                    and message.get("language")
                ):
                    render_spoken_answer(
                        message.get("content", ""),
                        voice_settings,
                        runtime_settings,
                        language=message["language"],
                        key_prefix=f"conversation_audio_{message_index}",
                    )
                if message["role"] == "assistant" and message.get("result"):
                    render_answer_quality(message["result"], message.get("min_score", 0.0))
                    feedback_key = message.get("feedback_key") or hashlib.sha256(
                        f"conversation|{message_index}|{message.get('time', '')}|{message.get('content', '')}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    with st.expander("Feedback", expanded=False):
                        render_feedback_controls(
                            feedback_key=f"conversation_{feedback_key}",
                            query=message.get("query", ""),
                            answer=message.get("content", ""),
                            result=message["result"],
                            runtime_settings=runtime_settings,
                            search_mode=message.get("search_mode", "hybrid"),
                            filters=message.get("filters", {}),
                            context="conversation",
                        )
                    with st.expander("Citations", expanded=False):
                        render_answer_sources(
                            message["result"],
                            message.get("min_score", 0.0),
                            runtime_settings,
                            key_prefix=f"conversation_{message_index}",
                        )

        prompt = st.chat_input("Message Enterprise RAG")
        prompt_to_send = voice_prompt.strip() if send_voice_prompt else (prompt or "").strip()
        if prompt_to_send:
            answer_language = response_language(voice_settings["language"], prompt_to_send)
            effective_filters = follow_up_filters(
                prompt_to_send,
                source_filters,
                latest_conversation_source_hashes(),
                store.documents,
            )
            augmented_prompt = conversation_context_prompt(prompt_to_send)
            use_follow_up_context = should_use_conversation_context(prompt_to_send, store.documents)
            retrieval_query = conversation_retrieval_query(prompt_to_send, store.documents)
            pinned_chunks = latest_conversation_citation_chunks(store) if use_follow_up_context else []
            st.session_state.conversation_messages.append(
                {"role": "user", "content": prompt_to_send, "language": answer_language}
            )
            with st.chat_message("assistant"):
                stream_placeholder = st.empty()
                with st.status("Retrieving context and streaming answer", expanded=False):
                    result = generate_rag_result(
                        query=augmented_prompt,
                        retrieval_query=retrieval_query,
                        chat_model=chat_model,
                        embedding_model=embedding_model,
                        top_k=top_k,
                        min_score=min_score,
                        response_language_name=answer_language,
                        filters=effective_filters,
                        search_mode=str(search_mode).lower(),
                        pinned_chunks=pinned_chunks,
                        stream_placeholder=stream_placeholder,
                    )
                stream_placeholder.markdown(result["answer"])
                render_answer_quality(result, min_score)
                render_spoken_answer(
                    result["answer"],
                    voice_settings,
                    runtime_settings,
                    language=answer_language,
                    key_prefix="conversation_latest",
                )

            st.session_state.conversation_messages.append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "language": answer_language,
                    "spoken_answer": voice_settings["spoken_answer"],
                    "feedback_key": hashlib.sha256(
                        f"{prompt_to_send}|{result.get('answer', '')}|{datetime.now().isoformat(timespec='seconds')}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                    "result": result,
                    "citations": result.get("source_metadata", []),
                    "model": chat_model,
                    "index": embedding_model,
                    "query": prompt_to_send,
                    "search_mode": str(search_mode).lower(),
                    "filters": effective_filters,
                    "min_score": min_score,
                    "time": datetime.now().isoformat(timespec="seconds"),
                }
            )
            st.rerun()


def render_agent() -> None:
    render_header("Agentic RAG", "Let the app choose a retrieval tool, reason over evidence, and produce a report.")
    embedding_model = model_selectbox(
        "Knowledge index",
        EMBEDDING_MODEL_OPTIONS,
        active_embedding_model(),
        "agent_embedding_model",
        disabled=not can_change_models(),
    )
    if not can_change_models():
        embedding_model = active_embedding_model()
    st.session_state.embedding_model = embedding_model

    store = get_vector_store(embedding_model)
    if store.is_empty:
        render_index_empty_state("Agent", "agent_empty_index")
        return

    st.markdown(
        """
        <div class="agent-tool-grid">
            <div class="agent-tool-card">
                <div class="agent-tool-title">Search</div>
                <div class="agent-tool-copy">Answer directly from top-k source chunks.</div>
            </div>
            <div class="agent-tool-card">
                <div class="agent-tool-title">Summarize</div>
                <div class="agent-tool-copy">Condense selected documents into key facts and risks.</div>
            </div>
            <div class="agent-tool-card">
                <div class="agent-tool-title">Compare</div>
                <div class="agent-tool-copy">Contrast two or more documents with cited differences.</div>
            </div>
            <div class="agent-tool-card">
                <div class="agent-tool-title">Report</div>
                <div class="agent-tool-copy">Generate a Markdown and PDF-ready evidence report.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    documents = store.list_documents()
    document_options = {document["file_hash"]: document.get("file_name", document["file_hash"]) for document in documents}

    with st.container(border=True, key="agent_settings"):
        with st.expander("Agent controls", expanded=False):
            model_col, tool_col, retrieval_col, voice_col = st.columns([1, 1, 1, 1], gap="large")
            with model_col:
                chat_model = model_selectbox(
                    "Answer model",
                    CHAT_MODEL_OPTIONS,
                    active_chat_model(),
                    "agent_chat_model",
                    disabled=not can_change_models(),
                )
                if not can_change_models():
                    chat_model = active_chat_model()
            with tool_col:
                requested_tool_label = st.selectbox(
                    "Agent tool",
                    ["Auto", "Search Documents", "Summarize Documents", "Compare Documents", "Generate Report"],
                    key="agent_requested_tool",
                )
                requested_tool = {
                    "Auto": "Auto",
                    "Search Documents": "search_documents",
                    "Summarize Documents": "summarize_documents",
                    "Compare Documents": "compare_documents",
                    "Generate Report": "generate_report",
                }[requested_tool_label]
                selected_documents = st.multiselect(
                    "Focus documents",
                    options=list(document_options.keys()),
                    format_func=lambda value: document_options.get(value, value),
                    key="agent_focus_documents",
                    help="Optional. Select documents for summarization, comparison, or focused search.",
                )
            with retrieval_col:
                search_mode = st.segmented_control(
                    "Search mode",
                    ["Hybrid", "Semantic", "Keyword"],
                    default="Hybrid",
                    key="agent_search_mode",
                )
                runtime_settings = active_settings(chat_model=chat_model, embedding_model=embedding_model)
                normalize_demo_top_k_state("agent_top_k", runtime_settings)
                top_k = st.slider(
                    "Top K",
                    1,
                    demo_top_k_limit(runtime_settings),
                    demo_top_k_value(runtime_settings),
                    key="agent_top_k",
                )
                min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, step=0.01, key="agent_min_score")
            with voice_col:
                st.subheader("Voice")
                voice_settings = render_voice_settings("agent", compact=True)

    runtime_settings = active_settings(
        chat_model=chat_model,
        embedding_model=embedding_model,
        transcription_model=voice_settings["transcription_model"],
        tts_model=voice_settings["tts_model"],
        tts_voice=voice_settings["tts_voice"],
    )

    render_voice_input(
        "agent",
        "Speak your agent goal",
        voice_settings,
        runtime_settings,
        target_text_key="agent_voice_review",
    )
    voice_goal = ""
    if st.session_state.get("agent_voice_review"):
        voice_goal = st.text_area(
            "Review voice goal",
            key="agent_voice_review",
            height=90,
        )
        st.caption("The agent will use the typed goal first, otherwise it will use this voice transcript.")

    goal = st.text_area(
        "Agent goal",
        key="agent_goal",
        height=110,
        placeholder="Example: Compare the Heidi and Black Beauty documents and generate a short report with citations.",
    )
    active_goal = goal.strip() or voice_goal.strip()
    inferred_focus_documents = infer_agent_document_hashes(active_goal, store.documents) if active_goal else []
    effective_focus_documents = unique_hashes(selected_documents + inferred_focus_documents)
    if inferred_focus_documents and not selected_documents:
        st.caption(f"Auto inferred focus documents: {describe_documents(inferred_focus_documents, document_options)}")

    compare_blocked = requested_tool == "compare_documents" and len(effective_focus_documents) < 2
    if compare_blocked:
        st.warning("Compare Documents needs at least two focus documents. Select them or mention their filenames in the goal.")

    run_col, clear_col = st.columns([1, 0.24], gap="small")
    run_clicked = run_col.button(
        "Run agent",
        type="primary",
        disabled=not active_goal or compare_blocked,
        width="stretch",
    )
    clear_col.button("Clear", width="stretch", on_click=reset_agent_workspace)

    if run_clicked:
        answer_language = response_language(voice_settings["language"], active_goal)
        with st.status("Planning tool call and gathering evidence", expanded=True):
            agent_result = run_agentic_rag(
                goal=active_goal,
                requested_tool=requested_tool,
                chat_model=chat_model,
                embedding_model=embedding_model,
                selected_hashes=selected_documents,
                top_k=top_k,
                min_score=min_score,
                search_mode=str(search_mode).lower(),
                response_language_name=answer_language,
            )
        agent_result["voice_settings"] = voice_settings
        agent_result["answer_language"] = answer_language
        st.session_state.last_agent_result = agent_result
        st.session_state.agent_history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "goal": active_goal,
                "tool": AGENT_TOOL_LABELS.get(agent_result["tool"], agent_result["tool"]),
                "confidence": agent_result["result"].get("confidence", 0.0),
                "sources": len(agent_result["result"].get("sources", [])),
            }
        )
        st.rerun()

    last = st.session_state.last_agent_result
    if not last:
        st.markdown(
            """
            <div class="conversation-empty-state">
                <div>
                    <div class="conversation-empty-title">What should the agent do?</div>
                    <div class="rag-subtle">Ask it to search, summarize, compare, or generate a cited report.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = last["result"]
    st.markdown(
        f"""
        <div class="agent-plan">
            <strong>Agent plan</strong><br>
            {escape_html(" | ".join(last["plan"]))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Agent Answer")
    st.markdown(last["answer"])
    render_answer_quality(result, min_score)
    render_spoken_answer(
        last["answer"],
        last.get("voice_settings", voice_settings),
        last["runtime_settings"],
        language=last.get("answer_language", response_language(voice_settings["language"], last["answer"])),
        key_prefix="agent_answer",
    )

    evidence_tab, report_tab, citations_tab, history_tab = st.tabs(
        ["Evidence", "Report", "Citations", "History"]
    )
    with evidence_tab:
        evidence_df = agent_evidence_dataframe(result)
        if evidence_df.empty:
            st.info("No evidence was retrieved for this agent run.")
        else:
            st.dataframe(evidence_df, hide_index=True, width="stretch")
            chart_df = evidence_df.groupby("Document", as_index=False)["Score"].max()
            fig = px.bar(
                chart_df,
                x="Document",
                y="Score",
                title="Top Evidence Score by Document",
                range_y=[0, 1],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=340)
            st.plotly_chart(fig, width="stretch")

    with report_tab:
        report_markdown = last["report_markdown"]
        st.markdown(report_markdown)
        export_col_a, export_col_b = st.columns(2)
        export_col_a.download_button(
            "Download Markdown report",
            report_markdown,
            "agentic_rag_report.md",
            "text/markdown",
            width="stretch",
        )
        export_col_b.download_button(
            "Download PDF report",
            agent_pdf_report(report_markdown),
            "agentic_rag_report.pdf",
            "application/pdf",
            width="stretch",
        )

    with citations_tab:
        render_answer_sources(
            result,
            min_score,
            last["runtime_settings"],
            key_prefix="agent",
        )

    with history_tab:
        if st.session_state.agent_history:
            st.dataframe(
                pd.DataFrame(list(reversed(st.session_state.agent_history))),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("No agent runs yet.")


def render_retrieval_audit() -> None:
    render_header("Retrieval Audit", "Inspect why chunks are selected before answer generation.")

    embedding_model = model_selectbox(
        "Knowledge index",
        EMBEDDING_MODEL_OPTIONS,
        active_embedding_model(),
        "audit_embedding_model",
        disabled=not can_change_models(),
    )
    if not can_change_models():
        embedding_model = active_embedding_model()
    st.session_state.embedding_model = embedding_model
    runtime_settings = active_settings(embedding_model=embedding_model)
    store = get_vector_store(embedding_model)

    if store.is_empty:
        render_index_empty_state("Retrieval Audit", "audit_empty_index")
        return

    st.markdown(
        """
        <div class="audit-guide-grid">
            <div class="audit-guide-card">
                <div class="audit-guide-title">Semantic match</div>
                <div class="audit-guide-copy">Compares the question vector with stored chunk vectors.</div>
            </div>
            <div class="audit-guide-card">
                <div class="audit-guide-title">Keyword match</div>
                <div class="audit-guide-copy">Rewards exact names, IDs, dates, and file terms.</div>
            </div>
            <div class="audit-guide-card">
                <div class="audit-guide-title">Hybrid rank</div>
                <div class="audit-guide-copy">Combines semantic and keyword evidence for final ordering.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    query = st.text_input("Search query")
    col_a, col_b, col_c = st.columns([1, 1, 1])
    search_mode = col_a.segmented_control(
        "Search mode",
        ["Hybrid", "Semantic", "Keyword"],
        default="Hybrid",
        key="audit_search_mode",
    )
    normalize_demo_top_k_state("audit_top_k", runtime_settings)
    top_k = col_b.slider(
        "Top K",
        min_value=1,
        max_value=demo_top_k_limit(runtime_settings),
        value=demo_top_k_value(runtime_settings),
        key="audit_top_k",
    )
    min_score = col_c.slider(
        "Minimum score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        key="audit_min_score",
    )

    if st.button("Run retrieval audit", type="primary", disabled=not query.strip()):
        audit_calls = 1 if str(search_mode).lower() in {"hybrid", "semantic"} else 0
        if audit_calls and not require_demo_budget(
            "Retrieval audit",
            estimated_calls=audit_calls,
            active_settings=runtime_settings,
        ):
            return
        with st.spinner("Searching vector index"):
            results = Retriever(store, runtime_settings).retrieve(
                query,
                top_k=top_k,
                search_mode=str(search_mode).lower(),
            )
        filtered = [item for item in results if float(item.get("score", 0.0)) >= min_score]

        if not filtered:
            st.info("No chunks met the selected threshold.")
            return

        rows = []
        for item in filtered:
            metadata = item.get("metadata", {})
            retrieval_method = item.get("retrieval_method") or metadata.get("retrieval_method") or str(search_mode).lower()
            rows.append(
                {
                    "Score": round(float(item.get("score", 0.0)), 4),
                    "Match": str(retrieval_method).title(),
                    "Document": metadata.get("file_name"),
                    "Page": metadata.get("page_number") or "",
                    "Chunk": metadata.get("chunk_index"),
                    "Why selected": retrieval_reason(retrieval_method),
                    "Preview": item.get("text", "")[:220],
                }
            )
        st.dataframe(rows, hide_index=True, width="stretch")

        for index, item in enumerate(filtered, start=1):
            metadata = item.get("metadata", {})
            retrieval_method = item.get("retrieval_method") or metadata.get("retrieval_method") or str(search_mode).lower()
            with st.expander(
                f"Match {index}: {metadata.get('file_name', 'Unknown')} - {str(retrieval_method).title()} - score {float(item.get('score', 0.0)):.2f}"
            ):
                st.caption(retrieval_reason(retrieval_method))
                st.write(item.get("text", ""))


def render_documents() -> None:
    render_header("Document Inventory", "Review indexed content for the active embedding model.")

    embedding_model = model_selectbox(
        "Knowledge index",
        EMBEDDING_MODEL_OPTIONS,
        active_embedding_model(),
        "documents_embedding_model",
        disabled=not can_change_models(),
    )
    if not can_change_models():
        embedding_model = active_embedding_model()
    st.session_state.embedding_model = embedding_model
    rows = document_rows(embedding_model)
    if not rows:
        render_index_empty_state("Documents", "documents_empty_index")
        return

    filter_text = st.text_input("Filter documents")
    if filter_text.strip():
        needle = filter_text.strip().lower()
        rows = [row for row in rows if needle in row["Document"].lower()]

    st.dataframe(rows, hide_index=True, width="stretch")
    csv = "\n".join(
        ["Document,Chunks,Visual chunks,Embedding model,Indexed at"]
        + [
            (
                f"\"{row['Document']}\",{row['Chunks']},{row['Visual chunks']},"
                f"\"{row['Embedding model']}\",\"{row['Indexed at']}\""
            )
            for row in rows
        ]
    )
    st.download_button("Export inventory", csv, "rag_document_inventory.csv", "text/csv")


def render_index_management() -> None:
    render_header("Index Management", "Delete, rebuild, re-index, and migrate FAISS indexes.")
    if not require_admin_ui():
        return

    embedding_model = model_selectbox(
        "Current embedding index",
        EMBEDDING_MODEL_OPTIONS,
        active_embedding_model(),
        "index_management_embedding_model",
    )
    st.session_state.embedding_model = embedding_model
    runtime_settings = active_settings(embedding_model=embedding_model)
    store = get_vector_store(embedding_model)
    documents = store.list_documents()

    if not documents:
        st.info("No documents indexed for this embedding model.")
        return

    document_map = {document["file_hash"]: document for document in documents}
    selected_hashes = st.multiselect(
        "Documents",
        options=list(document_map.keys()),
        format_func=lambda value: document_map[value].get("file_name", value),
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("Document Actions")
        if st.button("Delete selected documents", disabled=not selected_hashes, width="stretch"):
            for file_hash in selected_hashes:
                store.remove_document(file_hash)
            clear_index_caches()
            st.success(f"Deleted {len(selected_hashes)} document(s).")
            st.rerun()

        if st.button("Re-index selected documents", disabled=not selected_hashes, width="stretch"):
            successes = 0
            failures: list[str] = []
            for file_hash in selected_hashes:
                try:
                    reindex_document(document_map[file_hash], runtime_settings, force=True)
                    successes += 1
                except Exception as exc:
                    failures.append(f"{document_map[file_hash].get('file_name', file_hash)}: {exc}")
            clear_index_caches()
            if failures:
                st.warning(f"Re-indexed {successes}; {len(failures)} failed.")
                st.write("\n".join(failures))
            else:
                st.success(f"Re-indexed {successes} document(s).")

    with col_b:
        st.subheader("Model Migration")
        target_embedding_model = model_selectbox(
            "Target embedding model",
            EMBEDDING_MODEL_OPTIONS,
            "text-embedding-3-large",
            "migration_target_embedding_model",
        )
        target_settings = active_settings(embedding_model=target_embedding_model)
        force_migration = st.checkbox("Overwrite duplicates in target index")
        migrate_all = st.checkbox("Migrate all documents in current index")
        migration_hashes = list(document_map.keys()) if migrate_all else selected_hashes

        if st.button("Migrate documents", disabled=not migration_hashes, width="stretch"):
            successes = 0
            failures: list[str] = []
            for file_hash in migration_hashes:
                try:
                    migrate_document(
                        document_map[file_hash],
                        runtime_settings,
                        target_settings,
                        force=force_migration,
                    )
                    successes += 1
                except Exception as exc:
                    failures.append(f"{document_map[file_hash].get('file_name', file_hash)}: {exc}")
            clear_index_caches()
            if failures:
                st.warning(f"Migrated {successes}; {len(failures)} failed.")
                st.write("\n".join(failures))
            else:
                st.success(f"Migrated {successes} document(s) to {target_embedding_model}.")

    st.subheader("Index Rebuild")
    confirm_rebuild = st.checkbox("I understand this will rebuild the current index from stored uploads.")
    if st.button("Rebuild current index", disabled=not confirm_rebuild, type="primary"):
        rebuild_docs = list(documents)
        store.reset()
        successes = 0
        failures: list[str] = []
        for document in rebuild_docs:
            try:
                reindex_document(document, runtime_settings, force=False)
                successes += 1
            except Exception as exc:
                failures.append(f"{document.get('file_name', document.get('file_hash'))}: {exc}")
        clear_index_caches()
        if failures:
            st.warning(f"Rebuilt {successes}; {len(failures)} failed.")
            st.write("\n".join(failures))
        else:
            st.success(f"Rebuilt index with {successes} document(s).")

    confirm_reset = st.checkbox("I understand this will remove all vectors from the current index.")
    if st.button("Reset current index", disabled=not confirm_reset):
        store.reset()
        clear_index_caches()
        st.success("Current index reset.")
        st.rerun()


def render_evaluation() -> None:
    render_header("Evaluation", "Run a small answer-quality test set against the current RAG pipeline.")
    if not require_admin_ui():
        return

    embedding_model = model_selectbox(
        "Knowledge index",
        EMBEDDING_MODEL_OPTIONS,
        active_embedding_model(),
        "evaluation_embedding_model",
        disabled=not can_change_models(),
    )
    if not can_change_models():
        embedding_model = active_embedding_model()
    chat_model = model_selectbox(
        "Answer model",
        CHAT_MODEL_OPTIONS,
        active_chat_model(),
        "evaluation_chat_model",
        disabled=not can_change_models(),
    )
    if not can_change_models():
        chat_model = active_chat_model()

    runtime_settings = active_settings(chat_model=chat_model, embedding_model=embedding_model)
    store = get_vector_store(embedding_model)
    if store.is_empty:
        render_index_empty_state("Evaluation", "evaluation_empty_index")
        return

    st.caption(
        "Metrics are deterministic heuristics for demos: answer overlap with the expected answer, "
        "retrieval confidence, citation filename match, and unknown-answer behavior."
    )

    with st.container(border=True):
        st.subheader("Test set")
        if st.button("Reload starter cases", width="content"):
            st.session_state.evaluation_cases = default_evaluation_cases()
            st.session_state.last_evaluation_results = []
            save_evaluation_cases(st.session_state.evaluation_cases, runtime_settings)
            st.rerun()

        cases_df = pd.DataFrame(st.session_state.evaluation_cases)
        edited_cases = st.data_editor(
            cases_df,
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "Question": st.column_config.TextColumn("Question", required=True),
                "Expected answer": st.column_config.TextColumn("Expected answer"),
                "Expected unknown": st.column_config.CheckboxColumn("Expected unknown"),
                "Required citation contains": st.column_config.TextColumn(
                    "Required citation contains",
                    help="Optional filename/path text that should appear in at least one citation.",
                ),
            },
            key="evaluation_cases_editor",
        )

    control_col_a, control_col_b, control_col_c = st.columns([1, 1, 1], gap="large")
    with control_col_a:
        search_mode = st.segmented_control(
            "Search mode",
            ["Hybrid", "Semantic", "Keyword"],
            default="Hybrid",
            key="evaluation_search_mode",
        )
    with control_col_b:
        normalize_demo_top_k_state("evaluation_top_k", runtime_settings)
        top_k = st.slider(
            "Top K",
            1,
            demo_top_k_limit(runtime_settings),
            demo_top_k_value(runtime_settings),
            key="evaluation_top_k",
        )
    with control_col_c:
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, step=0.01, key="evaluation_min_score")

    cases = normalize_evaluation_cases(edited_cases.to_dict("records"))
    cases_to_run = cases
    demo_max_evaluation_cases = int(demo_setting(runtime_settings, "demo_max_evaluation_cases"))
    if demo_limits_enabled(runtime_settings) and demo_max_evaluation_cases > 0:
        cases_to_run = cases[:demo_max_evaluation_cases]
        if len(cases) > len(cases_to_run):
            st.info(
                f"Public demo mode will run the first {len(cases_to_run)} evaluation case(s). "
                f"Set RAG_DEMO_MAX_EVALUATION_CASES higher to allow more."
            )
    run_col, save_col, clear_col = st.columns([1, 0.35, 0.35], gap="small")
    run_clicked = run_col.button(
        "Run evaluation",
        type="primary",
        disabled=not cases,
        width="stretch",
    )
    if save_col.button("Save cases", disabled=not cases, width="stretch"):
        st.session_state.evaluation_cases = cases
        save_evaluation_cases(cases, runtime_settings)
        st.success("Evaluation test set saved.")
    if clear_col.button("Clear results", width="stretch"):
        st.session_state.last_evaluation_results = []
        st.rerun()

    if run_clicked:
        st.session_state.evaluation_cases = cases
        save_evaluation_cases(cases, runtime_settings)
        rows: list[dict] = []
        progress = st.progress(0)
        status = st.empty()
        for index, case in enumerate(cases_to_run, start=1):
            status.info(f"Evaluating {index} of {len(cases_to_run)}: {case['Question']}")
            try:
                result = generate_rag_result(
                    query=case["Question"],
                    chat_model=chat_model,
                    embedding_model=embedding_model,
                    top_k=top_k,
                    min_score=min_score,
                    response_language_name="English",
                    filters=None,
                    search_mode=str(search_mode).lower(),
                )
                rows.append(evaluate_result_row(case, result, search_mode=str(search_mode).lower()))
            except Exception as exc:
                logger.exception("Evaluation case failed: %s", exc)
                rows.append(evaluation_error_row(case, exc, search_mode=str(search_mode).lower()))
            progress.progress(index / len(cases_to_run))
        status.success(f"Evaluation completed for {len(cases_to_run)} case(s).")
        st.session_state.last_evaluation_results = rows

    results = st.session_state.last_evaluation_results
    if not results:
        st.markdown(
            """
            <div class="conversation-empty-state">
                <div>
                    <div class="conversation-empty-title">No evaluation run yet</div>
                    <div class="rag-subtle">Run the starter test set or edit it for your presentation documents.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    results_df = pd.DataFrame(results)
    non_error_df = results_df[results_df["Status"] != "Error"]
    citation_df = non_error_df[non_error_df["Citation correctness"] != "N/A"]
    metric_col_a, metric_col_b, metric_col_c, metric_col_d = st.columns(4)
    avg_retrieval = non_error_df["Retrieval score"].mean() if not non_error_df.empty else 0
    avg_quality = non_error_df["Answer quality"].mean() if not non_error_df.empty else 0
    citation_accuracy = (citation_df["Citation correctness"] == "Pass").mean() if not citation_df.empty else 0
    idk_accuracy = (
        (non_error_df["I don't know accuracy"] == "Pass").mean()
        if not non_error_df.empty
        else 0
    )
    metric_col_a.metric("Avg Retrieval Score", f"{avg_retrieval:.2f}")
    metric_col_b.metric("Avg Answer Quality", f"{avg_quality:.2f}")
    metric_col_c.metric("Citation Correctness", f"{citation_accuracy:.0%}" if not citation_df.empty else "N/A")
    metric_col_d.metric("I Don't Know Accuracy", f"{idk_accuracy:.0%}")

    st.dataframe(
        results_df[
            [
                "Status",
                "Question",
                "Retrieval score",
                "Answer quality",
                "Citation correctness",
                "I don't know accuracy",
                "Overall",
                "Sources",
                "Top source",
                "Error",
            ]
        ],
        hide_index=True,
        width="stretch",
    )

    chart_df = non_error_df.melt(
        id_vars=["Question"],
        value_vars=["Retrieval score", "Answer quality", "Overall"],
        var_name="Metric",
        value_name="Score",
    )
    if not chart_df.empty:
        fig = px.bar(
            chart_df,
            x="Question",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            title="Evaluation Scores by Question",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=360)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Answers and expected outputs"):
        for row in results:
            st.markdown(f"**{row['Question']}**")
            st.caption(f"Expected: {row['Expected']}")
            if row.get("Error"):
                st.error(row["Error"])
            else:
                st.write(row["Answer"])

    st.download_button(
        "Export evaluation CSV",
        results_df.to_csv(index=False),
        "rag_evaluation.csv",
        "text/csv",
        width="stretch",
    )
    st.download_button(
        "Export evaluation JSON",
        json.dumps(results, ensure_ascii=False, indent=2),
        "rag_evaluation.json",
        "application/json",
        width="stretch",
    )


def date_bucket(value: str | None) -> str:
    if not value:
        return "Unknown"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return "Unknown"


def usage_trend_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "Date": date_bucket(record.get("timestamp")),
            "Operation": record.get("operation") or "unknown",
            "Calls": 1,
            "Tokens": int(record.get("total_tokens") or 0),
        }
        for record in records
    ]
    if not rows:
        return pd.DataFrame(columns=["Date", "Operation", "Calls", "Tokens"])
    return pd.DataFrame(rows).groupby(["Date", "Operation"], as_index=False).sum()


def feedback_trend_dataframe(records: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "Date": date_bucket(record.get("submitted_at")),
            "Sentiment": record.get("sentiment") or "unknown",
            "Count": 1,
        }
        for record in records
    ]
    if not rows:
        return pd.DataFrame(columns=["Date", "Sentiment", "Count"])
    return pd.DataFrame(rows).groupby(["Date", "Sentiment"], as_index=False).sum()


def top_queried_document_counts(feedback_records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for item in st.session_state.query_history:
        documents = str(item.get("top_documents", "") or "")
        for document in [part.strip() for part in documents.split(";") if part.strip()]:
            counts[document] += 1
    for message in st.session_state.conversation_messages:
        for metadata in message.get("citations", []) or []:
            file_name = metadata.get("file_name") if isinstance(metadata, dict) else None
            if file_name:
                counts[str(file_name)] += 1
    last_agent_result = st.session_state.get("last_agent_result") or {}
    for metadata in last_agent_result.get("result", {}).get("source_metadata", []):
        file_name = metadata.get("file_name")
        if file_name:
            counts[str(file_name)] += 1
    for record in feedback_records:
        for metadata in record.get("source_metadata", []) or []:
            if isinstance(metadata, dict) and metadata.get("file_name"):
                counts[str(metadata["file_name"])] += 1
    return counts


def estimate_usage_cost(
    usage_records: list[dict],
    *,
    prompt_rate_per_million: float,
    completion_rate_per_million: float,
    input_unit_rate_per_million: float,
    output_unit_rate_per_million: float,
) -> float:
    prompt_tokens = sum(int(record.get("prompt_tokens") or 0) for record in usage_records)
    completion_tokens = sum(int(record.get("completion_tokens") or 0) for record in usage_records)
    input_units = sum(int(record.get("input_count") or 0) for record in usage_records)
    output_units = sum(int(record.get("output_count") or 0) for record in usage_records)
    return (
        (prompt_tokens / 1_000_000) * prompt_rate_per_million
        + (completion_tokens / 1_000_000) * completion_rate_per_million
        + (input_units / 1_000_000) * input_unit_rate_per_million
        + (output_units / 1_000_000) * output_unit_rate_per_million
    )


def render_admin_analytics_dashboard(runtime_settings) -> None:
    st.subheader("Admin Analytics Dashboard")
    store = get_vector_store(runtime_settings.openai_embedding_model)
    documents = store.list_documents()
    usage_records = load_usage(runtime_settings)
    feedback_records = load_feedback(runtime_settings)
    queue_status = Counter(item.get("status", "unknown") for item in st.session_state.ingestion_queue)
    failed_uploads = int(queue_status.get("failed", 0))
    if st.session_state.last_ingestion:
        failed_uploads = max(failed_uploads, int(st.session_state.last_ingestion.get("failed", 0) or 0))

    with st.expander("Cost estimator rates", expanded=False):
        st.caption("Use your approved pricing sheet here. Defaults are zero to avoid hardcoded stale vendor pricing.")
        rate_col_a, rate_col_b, rate_col_c, rate_col_d = st.columns(4)
        prompt_rate = rate_col_a.number_input("Prompt tokens $/1M", min_value=0.0, value=0.0, step=0.01)
        completion_rate = rate_col_b.number_input("Completion tokens $/1M", min_value=0.0, value=0.0, step=0.01)
        input_unit_rate = rate_col_c.number_input("Input units $/1M", min_value=0.0, value=0.0, step=0.01)
        output_unit_rate = rate_col_d.number_input("Output units $/1M", min_value=0.0, value=0.0, step=0.01)
    estimated_cost = estimate_usage_cost(
        usage_records,
        prompt_rate_per_million=prompt_rate,
        completion_rate_per_million=completion_rate,
        input_unit_rate_per_million=input_unit_rate,
        output_unit_rate_per_million=output_unit_rate,
    )

    metric_col_a, metric_col_b, metric_col_c, metric_col_d, metric_col_e = st.columns(5)
    metric_col_a.metric("Documents", len(documents))
    metric_col_b.metric("Chunks", len(store.chunks))
    metric_col_c.metric("Vectors", store.total_vectors)
    metric_col_d.metric("Failed Uploads", failed_uploads)
    metric_col_e.metric("Estimated Cost", f"${estimated_cost:.4f}")

    analytics_tab_a, analytics_tab_b, analytics_tab_c = st.tabs(["Index", "Usage", "Feedback"])
    with analytics_tab_a:
        if documents:
            index_rows = [
                {
                    "Document": document.get("file_name", "Unknown"),
                    "File type": Path(document.get("file_name", "")).suffix.lower().lstrip(".") or "unknown",
                    "Chunks": int(document.get("chunk_count") or 0),
                    "Visual chunks": int(document.get("visual_chunk_count") or 0),
                    "Uploaded": format_timestamp(document.get("uploaded_at")),
                }
                for document in documents
            ]
            index_df = pd.DataFrame(index_rows)
            file_type_df = index_df.groupby("File type", as_index=False)["Document"].count()
            fig = px.bar(file_type_df, x="File type", y="Document", title="Documents by File Type")
            fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=320)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(index_df, hide_index=True, width="stretch")
        else:
            st.caption("No indexed documents yet.")

        if st.session_state.ingestion_queue:
            queue_df = pd.DataFrame(queue_rows())
            st.dataframe(queue_df, hide_index=True, width="stretch")
        else:
            st.caption("No active ingestion queue.")

    with analytics_tab_b:
        if usage_records:
            operation_df = pd.DataFrame(
                [
                    {
                        "Operation": operation,
                        "Calls": count,
                        "Tokens": sum(
                            int(record.get("total_tokens") or 0)
                            for record in usage_records
                            if (record.get("operation") or "unknown") == operation
                        ),
                    }
                    for operation, count in Counter(record.get("operation") or "unknown" for record in usage_records).items()
                ]
            )
            fig = px.bar(operation_df, x="Operation", y="Calls", title="Calls by Operation")
            fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=320)
            st.plotly_chart(fig, width="stretch")

            trend_df = usage_trend_dataframe(usage_records)
            if not trend_df.empty:
                fig = px.line(trend_df, x="Date", y="Tokens", color="Operation", markers=True, title="Token Usage Trend")
                fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=320)
                st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No usage events recorded yet.")

    with analytics_tab_c:
        document_counts = top_queried_document_counts(feedback_records)
        if document_counts:
            top_docs_df = pd.DataFrame(
                [{"Document": document, "Mentions": count} for document, count in document_counts.most_common(10)]
            )
            fig = px.bar(top_docs_df, x="Mentions", y="Document", orientation="h", title="Top Queried/Cited Documents")
            fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=360)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(top_docs_df, hide_index=True, width="stretch")
        else:
            st.caption("No cited document activity yet. Ask questions or collect feedback to populate this chart.")

        trend_df = feedback_trend_dataframe(feedback_records)
        if not trend_df.empty:
            fig = px.bar(trend_df, x="Date", y="Count", color="Sentiment", title="Feedback Trend")
            fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=320)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No feedback submitted yet.")


def render_demo_limits_admin(runtime_settings) -> None:
    st.subheader("Public Demo Limits")
    if not demo_limits_enabled(runtime_settings):
        st.success("Demo usage limits are disabled for this deployment.")
        return

    status = demo_usage_status(runtime_settings)
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Session Calls", limit_text(status["session_calls"], status["session_call_limit"]))
    col_b.metric("Daily Calls", limit_text(status["daily_calls"], status["daily_call_limit"]))
    col_c.metric("Daily Tokens", limit_text(status["daily_tokens"], status["daily_token_limit"]))
    col_d.metric("Max Top K", int(demo_setting(runtime_settings, "demo_max_top_k")))
    st.caption(
        "Configured by RAG_DEMO_LIMITS_ENABLED, RAG_DEMO_SESSION_CALL_LIMIT, "
        "RAG_DEMO_DAILY_CALL_LIMIT, RAG_DEMO_DAILY_TOKEN_LIMIT, "
        "RAG_DEMO_MAX_UPLOAD_FILES, RAG_DEMO_MAX_UPLOAD_SIZE_MB, "
        "RAG_DEMO_MAX_TOP_K, RAG_DEMO_MAX_EVALUATION_CASES, "
        "RAG_DEMO_MAX_VISUAL_PAGES, and RAG_DEMO_MAX_DOCX_IMAGES."
    )
    if st.session_state.demo_blocked_actions:
        with st.expander("Blocked demo actions"):
            st.dataframe(
                list(reversed(st.session_state.demo_blocked_actions[-25:])),
                hide_index=True,
                width="stretch",
            )


def render_administration() -> None:
    render_header("Administration", "Runtime configuration, connectivity, and local storage state.")
    if not require_admin_ui():
        return

    render_metric_grid(active_embedding_model())
    runtime_settings = active_settings()

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.subheader("Models")
        st.write(f"Selected embedding index: `{runtime_settings.openai_embedding_model}`")
        st.write(f"Selected chat model: `{runtime_settings.openai_chat_model}`")
        st.write(f"Selected vision model: `{runtime_settings.openai_vision_model}`")
        st.write(f"Vision ingestion: `{runtime_settings.vision_ingestion_enabled}`")
        st.write(f"Vision detail: `{runtime_settings.vision_detail}`")
        st.write(f"Max answer tokens: `{runtime_settings.max_answer_tokens}`")
        st.write(f"Temperature: `{runtime_settings.openai_temperature}`")
        st.write(f"Available chat options: `{', '.join(CHAT_MODEL_OPTIONS)}`")
        st.write(f"Available embedding options: `{', '.join(EMBEDDING_MODEL_OPTIONS)}`")
        st.write(f"Available vision options: `{', '.join(VISION_MODEL_OPTIONS)}`")

    with col_b:
        st.subheader("Storage")
        st.write(f"Index directory: `{runtime_settings.index_dir}`")
        st.write(f"Upload directory: `{runtime_settings.upload_dir}`")
        st.write(f"Max files per batch: `{runtime_settings.max_upload_files}`")
        st.write(f"Per-file upload limit: `{format_size(runtime_settings.max_upload_size_mb)}`")
        if demo_limits_enabled(runtime_settings):
            st.write(f"Demo files per batch: `{demo_upload_file_limit(runtime_settings)}`")
            st.write(f"Demo per-file upload limit: `{format_size(demo_upload_size_limit_mb(runtime_settings))}`")

    st.subheader("Connectivity")
    st.write(f"SSL mode: `{ssl_runtime_description(settings)}`")
    if st.button("Run connection test", type="primary"):
        try:
            if require_demo_budget("OpenAI connection test", active_settings=runtime_settings):
                with st.status("Calling OpenAI embeddings API", expanded=False):
                    generate_embeddings(["enterprise rag connection test"], active_settings=runtime_settings)
                st.success("OpenAI connection verified.")
        except RAGApplicationError as exc:
            st.error(exc.message)

    render_demo_limits_admin(runtime_settings)
    render_admin_analytics_dashboard(runtime_settings)

    st.subheader("Usage Event Export")
    usage_records = load_usage(runtime_settings)
    usage_summary = summarize_usage(usage_records)
    usage_col_a, usage_col_b, usage_col_c, usage_col_d = st.columns(4)
    usage_col_a.metric("Tracked Calls", usage_summary["calls"])
    usage_col_b.metric("Tracked Tokens", usage_summary["tokens"])
    usage_col_c.metric("Tracked Documents", usage_summary["documents"])
    usage_col_d.metric(
        "Audio/Image Units",
        sum(int(record.get("input_count") or 0) + int(record.get("output_count") or 0) for record in usage_records),
    )

    if usage_records:
        operation_rows = []
        for operation in sorted(usage_summary["by_operation"]):
            operation_records = [record for record in usage_records if record.get("operation") == operation]
            operation_rows.append(
                {
                    "Operation": operation,
                    "Calls": len(operation_records),
                    "Tokens": sum(int(record.get("total_tokens") or 0) for record in operation_records),
                    "Input units": sum(int(record.get("input_count") or 0) for record in operation_records),
                    "Output units": sum(int(record.get("output_count") or 0) for record in operation_records),
                    "Documents": len({record.get("document_name") for record in operation_records if record.get("document_name")}),
                }
            )
        st.dataframe(operation_rows, hide_index=True, width="stretch")

        recent_usage_rows = [
            {
                "Time": format_timestamp(record.get("timestamp")),
                "Operation": record.get("operation"),
                "Model": record.get("model"),
                "Document": record.get("document_name") or "",
                "Tokens": record.get("total_tokens", 0),
                "Input units": record.get("input_count", 0),
                "Output units": record.get("output_count", 0),
            }
            for record in reversed(usage_records[-30:])
        ]
        with st.expander("Recent usage events"):
            st.dataframe(recent_usage_rows, hide_index=True, width="stretch")
        st.download_button(
            "Export usage JSONL",
            usage_jsonl(usage_records),
            "rag_usage.jsonl",
            "application/jsonl",
            width="stretch",
        )
        st.download_button(
            "Export usage CSV",
            usage_csv(usage_records),
            "rag_usage.csv",
            "text/csv",
            width="stretch",
        )
        st.caption("Dollar estimates are intentionally not hardcoded. Use the exported usage with your approved pricing sheet.")
    else:
        st.caption("No usage has been recorded yet. Embedding, vision, chat, speech, and transcription calls will appear here.")

    st.subheader("Feedback Export")
    feedback_records = load_feedback(runtime_settings)
    negative_count = sum(1 for record in feedback_records if record.get("sentiment") == "down")
    col_feedback_a, col_feedback_b, col_feedback_c = st.columns(3)
    col_feedback_a.metric("Feedback Records", len(feedback_records))
    col_feedback_b.metric("Bad Retrieval Flags", negative_count)
    col_feedback_c.metric("Useful Answers", sum(1 for record in feedback_records if record.get("sentiment") == "up"))

    if feedback_records:
        preview_rows = [
            {
                "Submitted": format_timestamp(record.get("submitted_at")),
                "Sentiment": record.get("sentiment"),
                "Query": str(record.get("query", ""))[:120],
                "Search": record.get("search_mode"),
                "Confidence": record.get("confidence"),
            }
            for record in reversed(feedback_records[-25:])
        ]
        st.dataframe(preview_rows, hide_index=True, width="stretch")
        st.download_button(
            "Export feedback JSONL",
            feedback_jsonl(feedback_records),
            "rag_feedback.jsonl",
            "application/jsonl",
            width="stretch",
        )
        st.download_button(
            "Export feedback CSV",
            feedback_csv(feedback_records),
            "rag_feedback.csv",
            "text/csv",
            width="stretch",
        )
    else:
        st.caption("No answer feedback has been submitted yet.")


def render_selected_page(selected: str) -> None:
    if selected == "Dashboard":
        render_dashboard()
    elif selected == "Ask":
        render_ask()
    elif selected == "Conversation":
        render_conversation()
    elif selected == "Agent":
        render_agent()
    elif selected == "Ingestion":
        render_ingestion()
    elif selected == "Retrieval Audit":
        render_retrieval_audit()
    elif selected == "Documents":
        render_documents()
    elif selected == "Index Management":
        render_index_management()
    elif selected == "Evaluation":
        render_evaluation()
    elif selected == "Administration":
        render_administration()


def main() -> None:
    st.set_page_config(page_title="Enterprise RAG", layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    inject_enterprise_styles()

    if settings.auth_enabled and not st.session_state.authenticated:
        render_login_page()
        return

    if not can_access_nav(st.session_state.nav_selection):
        st.session_state.nav_selection = default_nav_selection()

    navigation_mode = active_navigation_mode()
    selected = st.session_state.nav_selection
    render_top_bar(selected)
    if navigation_mode in {"Top row", "Both"}:
        render_workspace_nav(selected)
    render_demo_limit_status(active_settings())

    try:
        if navigation_mode in {"Sidebar", "Both"}:
            side_col, content_col = st.columns([0.22, 0.78], gap="large")
            with side_col:
                selected = render_app_sidebar(selected)
            with content_col:
                render_selected_page(selected)
        else:
            render_selected_page(selected)
    except RAGApplicationError as exc:
        st.error(exc.message)
    except Exception as exc:
        logger.exception("Unexpected UI failure: %s", exc)
        st.error("Unexpected error. Check the terminal logs for details.")


if __name__ == "__main__":
    main()
