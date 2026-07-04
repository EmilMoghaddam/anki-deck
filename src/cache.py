"""Disk cache for LLM responses."""

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)


def _cache_key(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(prefix: str, *key_parts: str) -> Path:
    """Return cache file path for given key parts."""
    key = _cache_key(*key_parts)
    return CACHE_DIR / f"{prefix}_{key}.json"


def get_cached(prefix: str, *key_parts: str) -> dict[str, Any] | None:
    """Load cached JSON if present."""
    path = cache_path(prefix, *key_parts)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Invalid cache file %s: %s", path, exc)
        return None


def set_cached(prefix: str, data: dict[str, Any], *key_parts: str) -> Path:
    """Atomically write data to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(prefix, *key_parts)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=CACHE_DIR,
        delete=False,
        suffix=".tmp",
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    logger.debug("Cached %s", path.name)
    return path


def invalidate_cached(prefix: str, *key_parts: str) -> bool:
    """Remove a cache entry."""
    path = cache_path(prefix, *key_parts)
    if path.exists():
        path.unlink()
        return True
    return False
