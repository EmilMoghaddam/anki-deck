"""Brazilian Portuguese TTS via Microsoft Edge (edge-tts)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path

import edge_tts
from tqdm import tqdm

from src.config import TTS_CACHE_DIR, TTS_CONCURRENCY, TTS_RATE, TTS_VOICE

logger = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"</?b>", re.IGNORECASE)


def strip_html(text: str) -> str:
    """Remove simple HTML tags before sending text to TTS."""
    return HTML_TAG_RE.sub("", text).strip()


def _cache_path(text: str, voice: str, rate: str) -> Path:
    key = hashlib.sha256(f"{voice}|{rate}|{text}".encode()).hexdigest()[:20]
    return TTS_CACHE_DIR / f"{key}.mp3"


def sound_field(filename: str) -> str:
    """Anki media reference for a bundled audio file."""
    return f"[sound:{filename}]"


async def _synthesize_one(
    text: str,
    dest: Path,
    voice: str,
    rate: str,
    sem: asyncio.Semaphore,
) -> None:
    async with sem:
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(dest))


async def _ensure_audio_async(
    items: list[tuple[str, Path]],
    voice: str,
    rate: str,
    concurrency: int,
) -> None:
    """Generate missing MP3 files for (text, path) pairs."""
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = [(text, path) for text, path in items if not path.exists()]
    if not pending:
        return

    sem = asyncio.Semaphore(concurrency)
    tasks = [_synthesize_one(text, path, voice, rate, sem) for text, path in pending]
    failed = 0
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="TTS audio"):
        try:
            await coro
        except Exception as exc:
            failed += 1
            logger.warning("TTS failed: %s", exc)
    if failed:
        logger.warning("TTS: %d/%d clips failed", failed, len(pending))


def ensure_audio(
    items: list[tuple[str, Path]],
    voice: str = TTS_VOICE,
    rate: str = TTS_RATE,
    concurrency: int = TTS_CONCURRENCY,
) -> None:
    """Sync wrapper for batch TTS generation."""
    asyncio.run(_ensure_audio_async(items, voice, rate, concurrency))


def audio_path_for_text(text: str, voice: str = TTS_VOICE, rate: str = TTS_RATE) -> Path:
    """Return cache path for a text clip (file may not exist yet)."""
    clean = strip_html(text)
    return _cache_path(clean, voice, rate)


def collect_card_audio_items(cards: list[dict]) -> list[tuple[str, Path]]:
    """Collect unique (text, cache_path) pairs needed for a card batch."""
    seen: set[Path] = set()
    items: list[tuple[str, Path]] = []

    def add(text: str) -> None:
        clean = strip_html(text)
        if not clean:
            return
        path = _cache_path(clean, TTS_VOICE, TTS_RATE)
        if path in seen:
            return
        seen.add(path)
        items.append((clean, path))

    for card in cards:
        add(card["lemma"])
        for ex in card.get("examples", []):
            add(ex.get("pt", ""))

    return items


def card_audio_fields(card: dict) -> dict[str, str]:
    """Return Anki [sound:…] field values for one card."""
    fields: dict[str, str] = {}

    lemma_path = audio_path_for_text(card["lemma"])
    if lemma_path.exists():
        fields["LemmaAudio"] = sound_field(lemma_path.name)
    else:
        fields["LemmaAudio"] = ""

    for i in range(1, 6):
        key = f"Ex{i}_Audio"
        examples = card.get("examples", [])
        if i - 1 < len(examples):
            path = audio_path_for_text(examples[i - 1].get("pt", ""))
            fields[key] = sound_field(path.name) if path.exists() else ""
        else:
            fields[key] = ""

    return fields
