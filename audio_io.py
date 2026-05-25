from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from config import Settings, settings
from errors import AudioProcessingError, clean_external_error
from openai_client import get_openai_client
from usage_store import record_usage


logger = logging.getLogger(__name__)


LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Portuguese": "pt",
}


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str


def transcribe_audio(
    audio_bytes: bytes,
    *,
    filename: str = "voice_input.wav",
    language: str = "Auto",
    active_settings: Settings = settings,
) -> TranscriptionResult:
    if not audio_bytes:
        raise AudioProcessingError("No audio was recorded.")

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    language_code = LANGUAGE_CODES.get(language)
    prompt = (
        "Transcribe enterprise RAG questions accurately. The user may speak in English, "
        "Hindi, or another major world language. Preserve the spoken language."
    )

    params = {
        "model": active_settings.openai_transcription_model,
        "file": audio_file,
        "response_format": "json",
        "prompt": prompt,
    }
    if language_code:
        params["language"] = language_code

    try:
        response = get_openai_client().audio.transcriptions.create(**params)
        transcript = response if isinstance(response, str) else getattr(response, "text", "")
        transcript = (transcript or "").strip()
        record_usage(
            "transcription",
            active_settings.openai_transcription_model,
            active_settings=active_settings,
            input_count=len(audio_bytes),
            output_count=len(transcript),
            metadata={"language": language},
        )
    except Exception as exc:
        raise AudioProcessingError(clean_external_error(exc, "Voice transcription")) from exc

    if not transcript:
        raise AudioProcessingError("Voice transcription returned no text.")

    resolved_language = language if language != "Auto" else infer_language(transcript)
    logger.info("Transcribed voice input as %s", resolved_language)
    return TranscriptionResult(text=transcript, language=resolved_language)


def synthesize_speech(
    text: str,
    *,
    language: str = "Auto",
    active_settings: Settings = settings,
) -> bytes:
    cleaned_text = (text or "").strip()
    if not cleaned_text:
        raise AudioProcessingError("No answer text was provided for speech generation.")

    instructions = _speech_instructions(language)
    try:
        response = get_openai_client().audio.speech.create(
            model=active_settings.openai_tts_model,
            voice=active_settings.openai_tts_voice,
            input=cleaned_text[:4096],
            instructions=instructions,
            response_format="mp3",
        )
        audio_bytes = response.read()
        record_usage(
            "speech",
            active_settings.openai_tts_model,
            active_settings=active_settings,
            input_count=len(cleaned_text),
            output_count=len(audio_bytes),
            metadata={"language": language, "voice": active_settings.openai_tts_voice},
        )
        return audio_bytes
    except Exception as exc:
        raise AudioProcessingError(clean_external_error(exc, "Voice output")) from exc


def infer_language(text: str) -> str:
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


def response_language(selected_language: str, text: str) -> str:
    if selected_language and selected_language != "Auto":
        return selected_language
    return infer_language(text)


def _speech_instructions(language: str) -> str:
    if language == "Hindi":
        return "Speak naturally in Hindi. Use clear pronunciation suitable for an enterprise demo."
    if language == "English":
        return "Speak naturally in clear English with a professional enterprise tone."
    if language and language != "Auto":
        return f"Speak naturally in {language} with a professional enterprise tone."
    return "Speak naturally in the same language as the input text with a professional enterprise tone."
