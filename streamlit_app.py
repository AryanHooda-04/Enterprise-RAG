from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from io import BytesIO
from collections import Counter
from datetime import datetime
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
    "Administration",
)
NAV_GROUPS = (
    ("Primary", PRIMARY_NAV_ITEMS),
    ("Knowledge", KNOWLEDGE_NAV_ITEMS),
    ("Admin tools", ADMIN_NAV_ITEMS),
)
NAV_ITEMS = PRIMARY_NAV_ITEMS + KNOWLEDGE_NAV_ITEMS + ADMIN_NAV_ITEMS

ADMIN_ONLY_NAV = {"Ingestion", "Index Management", "Administration"}
ROLES = ("Admin", "User")
NAVIGATION_MODES = ("Top row", "Sidebar", "Both")
COMPACT_NAV_LABELS = {
    "Retrieval Audit": "Audit",
    "Documents": "Docs",
    "Index Management": "Index",
    "Administration": "Admin",
}


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
    st.session_state.setdefault("feedback_submissions", {})
    st.session_state.setdefault("ingestion_queue", [])
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
        if st.sidebar.button("Sign out", use_container_width=True):
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
            padding-top: 4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 1.15, 1])

    with center:
        st.markdown(
            """
            <div class="login-card">
                <div class="login-title">Enterprise RAG Console</div>
                <div class="login-subtitle">Sign in to access the knowledge workspace.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="admin or user")
            password = st.text_input("Password", type="password", placeholder="admin or user")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)

        st.caption("Demo credentials: admin/admin or user/user")

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
    st.markdown(
        """
        <style>
        :root {
            --rag-bg: #0f141b;
            --rag-panel: #171d26;
            --rag-panel-2: #1f2631;
            --rag-border: #303846;
            --rag-text: #f5f7fa;
            --rag-muted: #a8b0bc;
            --rag-blue: #4f8cff;
            --rag-green: #2fbf71;
            --rag-amber: #d89b2b;
            --rag-red: #e05f5f;
        }

        .block-container {
            padding-top: 1.25rem;
            max-width: 1440px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
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
            background: linear-gradient(180deg, #121821 0%, #0f141b 100%);
        }

        .sidebar-brand {
            border: 1px solid var(--rag-border);
            background: rgba(79, 140, 255, 0.08);
            border-radius: 8px;
            padding: 0.9rem;
            margin: 0.25rem 0 0.85rem;
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
            background: var(--rag-panel);
            border-radius: 8px;
            padding: 1.4rem;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.24);
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
            border-color: var(--rag-border) !important;
            background: rgba(23, 29, 38, 0.94);
            backdrop-filter: blur(8px);
            border-radius: 8px;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.7rem;
        }

        .st-key-top_bar [data-testid="stHorizontalBlock"] {
            align-items: center;
        }

        .st-key-top_bar .stButton > button {
            width: 2.6rem;
            min-width: 2.6rem;
            min-height: 2.35rem;
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
        }

        .st-key-top_bar [data-testid="stSelectbox"] label {
            display: none;
        }

        .st-key-top_bar [data-baseweb="select"] > div {
            min-height: 2.1rem;
            border-radius: 999px;
            background: var(--rag-panel-2);
            border-color: var(--rag-border);
            font-size: 0.78rem;
            font-weight: 700;
            position: relative;
            padding-right: 2.1rem;
        }

        [data-baseweb="select"] > div {
            position: relative;
            padding-right: 2.1rem;
        }

        [data-baseweb="select"] > div::after {
            content: "";
            position: absolute;
            right: 0.85rem;
            top: 50%;
            width: 0.45rem;
            height: 0.45rem;
            border-right: 2px solid var(--rag-muted);
            border-bottom: 2px solid var(--rag-muted);
            transform: translateY(-65%) rotate(45deg);
            pointer-events: none;
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
        }

        .topbar-actions {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 0.4rem;
            flex-wrap: wrap;
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
            margin-bottom: 1.5rem;
            padding: 0.45rem 0.6rem;
            border-color: var(--rag-border) !important;
            background: rgba(23, 29, 38, 0.72);
            border-radius: 8px;
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

        .role-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid var(--rag-border);
            background: var(--rag-panel-2);
            color: var(--rag-text);
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.25;
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
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
            background: transparent;
            color: var(--rag-muted);
        }

        [data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
            color: var(--rag-text);
            background: rgba(255, 255, 255, 0.05);
            border-color: var(--rag-border);
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: rgba(79, 140, 255, 0.18);
            border-color: rgba(79, 140, 255, 0.45);
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
            border-bottom: 1px solid var(--rag-border);
            padding-bottom: 0.75rem;
            margin-bottom: 1.2rem;
        }

        .rag-title h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.1;
        }

        .rag-subtle {
            color: var(--rag-muted);
            font-size: 0.9rem;
        }

        .status-dot {
            display: inline-block;
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 999px;
            background: var(--rag-green);
            margin-right: 0.4rem;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 1rem;
        }

        .metric-card {
            border: 1px solid var(--rag-border);
            background: var(--rag-panel);
            border-radius: 8px;
            padding: 0.85rem 1rem;
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
            background: var(--rag-panel);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0 1rem;
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
            background: rgba(23, 29, 38, 0.52);
            border-radius: 8px;
            padding: 1.2rem;
            margin: 0.75rem 0 1rem;
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
            background: rgba(23, 29, 38, 0.64);
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
            background: rgba(23, 29, 38, 0.58);
            border-radius: 8px;
            padding: 0.85rem;
            margin: 0.65rem 0 0.35rem;
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
            background: rgba(23, 29, 38, 0.58);
            border-radius: 8px;
            padding: 0.75rem;
            min-height: 5.1rem;
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
            background: rgba(23, 29, 38, 0.34);
            padding: 2rem;
            margin: 1rem 0;
        }

        .conversation-empty-title {
            color: var(--rag-text);
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }

        .st-key-conversation_settings,
        .st-key-ask_settings {
            border-color: var(--rag-border) !important;
            background: rgba(23, 29, 38, 0.72);
            border-radius: 8px;
            margin-bottom: 0.9rem;
        }

        .st-key-conversation_chat_shell [data-testid="stChatMessage"],
        .st-key-ask_chat_shell [data-testid="stChatMessage"] {
            border-bottom: 0;
            padding: 1rem 0.25rem;
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
            background: rgba(23, 29, 38, 0.58);
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
            background: rgba(23, 29, 38, 0.58);
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
            background: var(--rag-panel-2);
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
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--rag-border);
            background: var(--rag-panel);
            border-radius: 8px;
            padding: 0.75rem;
        }

        @media (max-width: 900px) {
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
            }
            .answer-quality,
            .source-card-header {
                align-items: flex-start;
                flex-direction: column;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get("theme_mode") == "Light":
        st.markdown(
            """
            <style>
            :root {
                --rag-bg: #f6f8fb;
                --rag-panel: #ffffff;
                --rag-panel-2: #eef2f7;
                --rag-border: #d7dee9;
                --rag-text: #17202c;
                --rag-muted: #617086;
                --rag-blue: #245fd6;
                --rag-green: #158554;
                --rag-amber: #9a6a09;
                --rag-red: #b42323;
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
                background: rgba(255, 255, 255, 0.94);
            }
            .empty-state-panel,
            .answer-quality,
            .evidence-chip,
            .source-card,
            .ingestion-step,
            .audit-guide-card,
            .agent-tool-card {
                background: rgba(255, 255, 255, 0.86);
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
            <div class="rag-subtle"><span class="status-dot"></span>Local index online</div>
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
        with st.spinner("Generating voice answer"):
            cache[cache_key] = synthesize_speech(
                answer,
                language=language,
                active_settings=runtime_settings,
            )

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
        if st.button("Good answer", key=f"{feedback_key}_up", use_container_width=True):
            submit("up")
    with col_b:
        if st.button("Bad retrieval", key=f"{feedback_key}_down", use_container_width=True):
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
        menu_col, breadcrumb_col, actions_col = st.columns([0.09, 0.61, 0.3], gap="small")
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
        use_container_width=True,
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
                if st.button("Sign out", key="workspace_sign_out", use_container_width=True):
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
            if st.button("Sign out", key="app_sidebar_sign_out", use_container_width=True):
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
                    use_container_width=True,
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

        if st.button("Refresh index", key="app_side_refresh", use_container_width=True):
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
    return [
        {"Field": "Document", "Value": metadata.get("file_name") or "Unknown"},
        {"Field": "Page", "Value": metadata.get("page_number") or "N/A"},
        {"Field": "Source type", "Value": metadata.get("source_type") or "text"},
        {"Field": "Image", "Value": metadata.get("image_index") or "N/A"},
        {"Field": "Chunk", "Value": metadata.get("chunk_index")},
        {"Field": "Similarity", "Value": metadata.get("score")},
        {"Field": "Token start", "Value": metadata.get("token_start")},
        {"Field": "Token count", "Value": metadata.get("token_count")},
        {"Field": "Embedding model", "Value": metadata.get("embedding_model")},
        {"Field": "Retrieval", "Value": metadata.get("retrieval_method") or "semantic"},
        {"Field": "Semantic score", "Value": metadata.get("semantic_score")},
        {"Field": "Keyword score", "Value": metadata.get("keyword_score")},
    ]


@st.cache_data(show_spinner=False)
def pdf_page_preview_bytes(path_text: str, page_number: int, modified_at: float) -> bytes:
    del modified_at
    import fitz

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
            st.image(preview, caption=f"{source_path.name} - page {page_number}", use_container_width=True)
        except Exception as exc:
            st.caption(f"PDF preview unavailable: {exc}")
        return

    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        st.image(str(source_path), caption=source_path.name, use_container_width=True)
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
            response_language_name="English",
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
    for delta in pipeline.stream_answer(pipeline.build_prompt(agent_prompt, chunks, response_language="English")):
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


def persist_uploaded_file(uploaded_file, runtime_settings) -> tuple[Path, str, str]:
    original_name = safe_filename(uploaded_file.name)
    extension = Path(original_name).suffix.lower()

    if extension not in runtime_settings.allowed_extensions:
        allowed = ", ".join(runtime_settings.allowed_extensions)
        raise RAGApplicationError(f"Unsupported file type '{extension}'. Allowed: {allowed}.")

    runtime_settings.ensure_directories()
    temp_path = runtime_settings.upload_dir / f"pending_{uuid4().hex}_{original_name}"
    max_bytes = runtime_settings.max_upload_size_mb * 1024 * 1024
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
                    f"{original_name} exceeds the {format_size(runtime_settings.max_upload_size_mb)} per-file limit."
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
    for uploaded_file in uploaded_files or []:
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
                use_container_width=True,
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
    st.sidebar.divider()

    if st.sidebar.button("Refresh local index", use_container_width=True):
        get_pipeline.clear()
        get_vector_store.clear()
        st.rerun()

    if st.sidebar.button("Test OpenAI connection", use_container_width=True):
        try:
            with st.sidebar.status("Calling embeddings API", expanded=False):
                generate_embeddings(["connection test"], active_settings=active_settings())
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
    if action_col_a.button("Ask a question", key="dashboard_go_ask", type="primary", use_container_width=True):
        st.session_state.nav_selection = "Ask"
        st.rerun()
    if action_col_b.button("Open conversation", key="dashboard_go_conversation", use_container_width=True):
        st.session_state.nav_selection = "Conversation"
        st.rerun()
    if is_admin() and action_col_c.button("Upload documents", key="dashboard_go_ingestion", use_container_width=True):
        st.session_state.nav_selection = "Ingestion"
        st.rerun()

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader("Indexed Documents")
        rows = document_rows()
        if rows:
            st.dataframe(rows[:8], hide_index=True, use_container_width=True)
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
            f"{selected_count} selected. Batch limit: {settings.max_upload_files}. "
            f"Per-file limit: {format_size(settings.max_upload_size_mb)}."
        )

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
    start_disabled = not uploaded_files or len(uploaded_files) > runtime_settings.max_upload_files
    if uploaded_files and len(uploaded_files) > runtime_settings.max_upload_files:
        st.error(f"Select {runtime_settings.max_upload_files} documents or fewer per batch.")

    upload_action_col, _ = st.columns([1, 2])
    if upload_action_col.button("Add to queue", type="primary", disabled=start_disabled, use_container_width=True):
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
    if queue_col_b.button("Start queue", disabled=queued_count == 0, use_container_width=True):
        summary = process_ingestion_queue(runtime_settings)
        summary = {
            "time": datetime.now().strftime("%H:%M:%S"),
            **summary,
        }
        st.session_state.last_ingestion = summary

    if queue_col_c.button("Retry failed", disabled=failed_count == 0, use_container_width=True):
        for item in st.session_state.ingestion_queue:
            if item.get("status") == "failed":
                item["status"] = "queued"
                item["message"] = "Queued for retry"
        st.rerun()

    if st.session_state.ingestion_queue:
        st.subheader("Processing Queue")
        st.dataframe(queue_rows(), hide_index=True, use_container_width=True)
        clear_col_a, clear_col_b = st.columns([1, 3])
        if clear_col_a.button("Clear completed", use_container_width=True):
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
                st.dataframe(source_metadata_rows(metadata), hide_index=True, use_container_width=True)
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
                    use_container_width=True,
                )

                source_path = resolve_document_path(metadata.get("file_hash"), runtime_settings)
                if source_path and source_path.exists():
                    st.link_button(
                        "Open stored document",
                        source_path.resolve().as_uri(),
                        use_container_width=True,
                    )
                    st.download_button(
                        "Download stored document",
                        source_path.read_bytes(),
                        source_path.name,
                        key=f"{key_prefix}_document_download_{index}",
                        use_container_width=True,
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
                    top_k = st.slider(
                        "Top K",
                        min_value=1,
                        max_value=20,
                        value=base_runtime_settings.top_k,
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
            if st.button("New chat", key="ask_new_chat", use_container_width=True):
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
                use_container_width=True,
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
                }
            )
            st.rerun()

        if st.session_state.query_history:
            with st.expander("Query history"):
                st.dataframe(
                    list(reversed(st.session_state.query_history)),
                    hide_index=True,
                    use_container_width=True,
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
                    top_k = st.slider(
                        "Top K",
                        min_value=1,
                        max_value=20,
                        value=base_runtime_settings.top_k,
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
            if st.button("New chat", key="conversation_new_chat", use_container_width=True):
                reset_conversation_chat()
                st.rerun()
        if st.session_state.conversation_messages:
            with action_export_md:
                st.download_button(
                    "Markdown",
                    conversation_export_markdown(),
                    "rag_conversation.md",
                    "text/markdown",
                    use_container_width=True,
                )
            with action_export_json:
                st.download_button(
                    "JSON",
                    json.dumps(st.session_state.conversation_messages, indent=2),
                    "rag_conversation.json",
                    "application/json",
                    use_container_width=True,
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
                use_container_width=True,
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
            model_col, tool_col, retrieval_col = st.columns([1, 1, 1], gap="large")
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
                top_k = st.slider("Top K", 1, 20, runtime_settings.top_k, key="agent_top_k")
                min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, step=0.01, key="agent_min_score")

    goal = st.text_area(
        "Agent goal",
        key="agent_goal",
        height=110,
        placeholder="Example: Compare the Heidi and Black Beauty documents and generate a short report with citations.",
    )
    inferred_focus_documents = infer_agent_document_hashes(goal, store.documents) if goal.strip() else []
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
        disabled=not goal.strip() or compare_blocked,
        use_container_width=True,
    )
    if clear_col.button("Clear", use_container_width=True):
        st.session_state.last_agent_result = None
        st.rerun()

    if run_clicked:
        with st.status("Planning tool call and gathering evidence", expanded=True):
            agent_result = run_agentic_rag(
                goal=goal.strip(),
                requested_tool=requested_tool,
                chat_model=chat_model,
                embedding_model=embedding_model,
                selected_hashes=selected_documents,
                top_k=top_k,
                min_score=min_score,
                search_mode=str(search_mode).lower(),
            )
        st.session_state.last_agent_result = agent_result
        st.session_state.agent_history.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "goal": goal.strip(),
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

    evidence_tab, report_tab, citations_tab, history_tab = st.tabs(
        ["Evidence", "Report", "Citations", "History"]
    )
    with evidence_tab:
        evidence_df = agent_evidence_dataframe(result)
        if evidence_df.empty:
            st.info("No evidence was retrieved for this agent run.")
        else:
            st.dataframe(evidence_df, hide_index=True, use_container_width=True)
            chart_df = evidence_df.groupby("Document", as_index=False)["Score"].max()
            fig = px.bar(
                chart_df,
                x="Document",
                y="Score",
                title="Top Evidence Score by Document",
                range_y=[0, 1],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=46, b=10), height=340)
            st.plotly_chart(fig, use_container_width=True)

    with report_tab:
        report_markdown = last["report_markdown"]
        st.markdown(report_markdown)
        export_col_a, export_col_b = st.columns(2)
        export_col_a.download_button(
            "Download Markdown report",
            report_markdown,
            "agentic_rag_report.md",
            "text/markdown",
            use_container_width=True,
        )
        export_col_b.download_button(
            "Download PDF report",
            agent_pdf_report(report_markdown),
            "agentic_rag_report.pdf",
            "application/pdf",
            use_container_width=True,
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
                use_container_width=True,
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
    top_k = col_b.slider("Top K", min_value=1, max_value=20, value=runtime_settings.top_k, key="audit_top_k")
    min_score = col_c.slider(
        "Minimum score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        key="audit_min_score",
    )

    if st.button("Run retrieval audit", type="primary", disabled=not query.strip()):
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
        st.dataframe(rows, hide_index=True, use_container_width=True)

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

    st.dataframe(rows, hide_index=True, use_container_width=True)
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
        if st.button("Delete selected documents", disabled=not selected_hashes, use_container_width=True):
            for file_hash in selected_hashes:
                store.remove_document(file_hash)
            clear_index_caches()
            st.success(f"Deleted {len(selected_hashes)} document(s).")
            st.rerun()

        if st.button("Re-index selected documents", disabled=not selected_hashes, use_container_width=True):
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

        if st.button("Migrate documents", disabled=not migration_hashes, use_container_width=True):
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

    st.subheader("Connectivity")
    st.write(f"SSL mode: `{ssl_runtime_description(settings)}`")
    if st.button("Run connection test", type="primary"):
        try:
            with st.status("Calling OpenAI embeddings API", expanded=False):
                generate_embeddings(["enterprise rag connection test"], active_settings=runtime_settings)
            st.success("OpenAI connection verified.")
        except RAGApplicationError as exc:
            st.error(exc.message)

    st.subheader("Usage & Cost Dashboard")
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
        st.dataframe(operation_rows, hide_index=True, use_container_width=True)

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
            st.dataframe(recent_usage_rows, hide_index=True, use_container_width=True)
        st.download_button(
            "Export usage JSONL",
            usage_jsonl(usage_records),
            "rag_usage.jsonl",
            "application/jsonl",
            use_container_width=True,
        )
        st.download_button(
            "Export usage CSV",
            usage_csv(usage_records),
            "rag_usage.csv",
            "text/csv",
            use_container_width=True,
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
        st.dataframe(preview_rows, hide_index=True, use_container_width=True)
        st.download_button(
            "Export feedback JSONL",
            feedback_jsonl(feedback_records),
            "rag_feedback.jsonl",
            "application/jsonl",
            use_container_width=True,
        )
        st.download_button(
            "Export feedback CSV",
            feedback_csv(feedback_records),
            "rag_feedback.csv",
            "text/csv",
            use_container_width=True,
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
