from __future__ import annotations

from functools import lru_cache

import httpx
from openai import OpenAI

from config import Settings, settings, ssl_verify_for_httpx
from errors import ConfigurationError


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    return build_openai_client(settings)


def build_openai_client(active_settings: Settings) -> OpenAI:
    if not active_settings.openai_api_key:
        raise ConfigurationError("OPENAI_API_KEY is not set.")

    verify = ssl_verify_for_httpx(active_settings)

    http_client = httpx.Client(
        verify=verify,
        timeout=httpx.Timeout(active_settings.openai_timeout_seconds),
        trust_env=True,
    )

    return OpenAI(
        api_key=active_settings.openai_api_key,
        timeout=active_settings.openai_timeout_seconds,
        max_retries=active_settings.openai_max_retries,
        http_client=http_client,
    )
