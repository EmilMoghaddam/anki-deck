"""Shared OpenRouter client."""

import os

from openai import OpenAI

from src.config import OPENROUTER_APP_NAME, OPENROUTER_BASE_URL, OPENROUTER_SITE_URL


def get_client() -> OpenAI:
    """Create OpenRouter OpenAI-compatible client."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and add your key."
        )
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        default_headers={
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        },
    )
