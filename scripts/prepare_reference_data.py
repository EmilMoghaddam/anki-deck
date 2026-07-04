#!/usr/bin/env python3
"""Prepare seed PT→ES dictionary for cognate filtering."""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PT_ES_DICT_FILE, REFERENCE_DIR  # noqa: E402
from src.seed_dict import get_seed_dict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Write seed PT→ES dictionary."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    seed = get_seed_dict()
    PT_ES_DICT_FILE.write_text(
        json.dumps(dict(sorted(seed.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote seed PT-ES dict with %d entries to %s", len(seed), PT_ES_DICT_FILE)


if __name__ == "__main__":
    main()
