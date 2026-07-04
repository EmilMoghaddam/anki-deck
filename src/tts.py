"""Brazilian Portuguese TTS via Microsoft Edge (edge-tts)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import subprocess
import tempfile
from pathlib import Path

import edge_tts
import imageio_ffmpeg
from tqdm import tqdm

from src.config import TTS_CACHE_DIR, TTS_CONCURRENCY, TTS_RATE, TTS_VOICE

logger = logging.getLogger(__name__)

HTML_TAG_RE = re.compile(r"</?b>", re.IGNORECASE)

# Clips smaller than this are treated as failed TTS (Edge sometimes writes empty files).
MIN_AUDIO_BYTES = 512


def strip_html(text: str) -> str:
    """Remove simple HTML tags before sending text to TTS."""
    return HTML_TAG_RE.sub("", text).strip()


def _cache_path(text: str, voice: str, rate: str) -> Path:
    key = hashlib.sha256(f"{voice}|{rate}|{text}".encode()).hexdigest()[:20]
    return TTS_CACHE_DIR / f"{key}.mp3"


def _ffmpeg_exe() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def is_valid_audio(path: Path) -> bool:
    """True when a cached clip exists and looks like a usable MP3 for Anki."""
    if not path.exists():
        return False
    if path.stat().st_size < MIN_AUDIO_BYTES:
        return False
    with open(path, "rb") as f:
        header = f.read(3)
    return header == b"ID3" or header[:2] == b"\xff\xfb" or header[:2] == b"\xff\xf3"


def invalidate_audio(path: Path) -> None:
    """Remove a corrupt or empty cache file so it can be regenerated."""
    if path.exists():
        path.unlink()


def needs_transcode(path: Path) -> bool:
    """Edge TTS writes raw ADTS MP3; AnkiMobile needs ID3-framed MP3."""
    if not is_valid_audio(path):
        return False
    with open(path, "rb") as f:
        return f.read(3) != b"ID3"


def transcode_for_anki(path: Path) -> None:
    """Rewrite Edge TTS output as 44.1 kHz MP3 with ID3 header (iOS-compatible)."""
    if not needs_transcode(path):
        return
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=path.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                _ffmpeg_exe(),
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-acodec",
                "libmp3lame",
                "-ar",
                "44100",
                "-ac",
                "1",
                "-b:a",
                "64k",
                str(tmp_path),
            ],
            check=True,
        )
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def ensure_anki_compatible(paths: list[Path]) -> None:
    """Transcode any cached ADTS clips before bundling into .apkg."""
    pending = [path for path in paths if needs_transcode(path)]
    if not pending:
        return
    logger.info("Transcoding %d clips for AnkiMobile compatibility", len(pending))
    failed = 0
    for path in tqdm(pending, desc="Transcode audio"):
        try:
            transcode_for_anki(path)
        except Exception as exc:
            failed += 1
            logger.warning("Transcode failed for %s: %s", path.name, exc)
    if failed:
        logger.warning("Transcode: %d/%d clips failed", failed, len(pending))


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
        dest.unlink(missing_ok=True)
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(str(dest))
        if dest.stat().st_size < MIN_AUDIO_BYTES:
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"empty audio for {text[:60]!r}")
        transcode_for_anki(dest)
        if not is_valid_audio(dest):
            dest.unlink(missing_ok=True)
            raise RuntimeError(f"invalid audio after transcode for {text[:60]!r}")


async def _ensure_audio_async(
    items: list[tuple[str, Path]],
    voice: str,
    rate: str,
    concurrency: int,
) -> None:
    """Generate missing or corrupt MP3 files for (text, path) pairs."""
    TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    for text, path in items:
        if is_valid_audio(path):
            continue
        invalidate_audio(path)
        pending.append((text, path))
    if not pending:
        return

    sem = asyncio.Semaphore(concurrency)
    max_attempts = 3
    still_pending = pending

    for attempt in range(1, max_attempts + 1):
        if not still_pending:
            break
        if attempt > 1:
            logger.info(
                "Retrying %d TTS clips (attempt %d/%d)",
                len(still_pending),
                attempt,
                max_attempts,
            )
            await asyncio.sleep(2 * attempt)

        async def run_one(text: str, path: Path) -> tuple[tuple[str, Path], Exception | None]:
            try:
                await _synthesize_one(text, path, voice, rate, sem)
                return (text, path), None
            except Exception as exc:
                return (text, path), exc

        results = await asyncio.gather(*(run_one(text, path) for text, path in still_pending))
        failed_items: list[tuple[str, Path]] = []
        for item, exc in results:
            if exc is not None:
                failed_items.append(item)
                logger.warning("TTS failed: %s", exc)
        still_pending = failed_items

    if still_pending:
        logger.warning(
            "TTS: %d/%d clips failed after retries",
            len(still_pending),
            len(pending),
        )


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
