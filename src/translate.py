"""PT→ES translation for translation-first cognate filtering."""

import json
import logging
import re
import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from src.cache import get_cached, set_cached
from src.config import (
    CLEANED_PARQUET,
    MAX_RETRIES,
    OPENROUTER_MODEL,
    PT_ES_DICT_FILE,
    TEMPERATURE,
    TRANSLATE_BATCH_SIZE,
    TRANSLATE_PROMPT_VERSION,
    TRANSLATE_TOP_N,
)
from src.openrouter_client import get_client

logger = logging.getLogger(__name__)

POS_LABELS = {
    "nom": "noun",
    "ver": "verb",
    "adj": "adjective",
    "adv": "adverb",
}

SPANISH_GLOSS_RE = re.compile(r"^[a-záéíóúüñA-ZÁÉÍÓÚÜÑ\-]+$")

TRANSLATE_SYSTEM_PROMPT = """You are a Portuguese-to-Spanish lexicographer.

Translate Brazilian Portuguese lemmas into Spanish dictionary-form glosses.
Output valid JSON only, no markdown.

Rules:
- Latin American Spanish
- Dictionary forms (lemmas), never inflected forms
- 1-3 concise glosses per word
- Respect the part of speech given"""


class TranslationItem(BaseModel):
    lemma: str
    pos: str = ""
    spanish: list[str] = Field(min_length=1, max_length=3)

    @field_validator("spanish", mode="before")
    @classmethod
    def coerce_spanish(cls, value: object) -> object:
        if isinstance(value, str):
            return [value]
        return value


class TranslationBatchResponse(BaseModel):
    translations: list[TranslationItem]

    @classmethod
    def from_json(cls, data: object) -> "TranslationBatchResponse":
        if isinstance(data, list):
            return cls(translations=[TranslationItem.model_validate(item) for item in data])
        if isinstance(data, dict):
            entries = data.get("translations", data.get("results", []))
            if isinstance(entries, list):
                return cls(
                    translations=[TranslationItem.model_validate(item) for item in entries]
                )
        raise ValueError("Expected JSON object with translations array")


def _lemma_cache_key(lemma: str, pos: str) -> tuple[str, ...]:
    return (lemma.lower(), pos, TRANSLATE_PROMPT_VERSION, OPENROUTER_MODEL)


def _batch_cache_key(lemmas: list[tuple[str, str]]) -> tuple[str, ...]:
    payload = "|".join(f"{l}:{p}" for l, p in sorted(lemmas))
    return (payload, TRANSLATE_PROMPT_VERSION, OPENROUTER_MODEL)


def load_pt_es_dict(path: Path | None = None) -> dict[str, list[str]]:
    """Load PT→ES dictionary from JSON."""
    ref_path = path or PT_ES_DICT_FILE
    if not ref_path.exists():
        return {}
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in data.items()}


def save_pt_es_dict(data: dict[str, list[str]], path: Path | None = None) -> Path:
    """Persist PT→ES dictionary."""
    out = path or PT_ES_DICT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    sorted_data = dict(sorted(data.items()))
    out.write_text(json.dumps(sorted_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote PT-ES dict with %d entries to %s", len(sorted_data), out)
    return out


def _validate_spanish_gloss(gloss: str) -> str | None:
    """Return cleaned gloss or None if invalid."""
    gloss = gloss.strip().lower().split(",")[0].split(";")[0].strip()
    if not gloss or len(gloss) < 2:
        return None
    if not SPANISH_GLOSS_RE.match(gloss):
        return None
    return gloss


def _build_batch_prompt(items: list[tuple[str, str]]) -> str:
    lines = []
    for lemma, pos in items:
        pos_label = POS_LABELS.get(pos, pos)
        lines.append(f"- {lemma} ({pos_label})")
    word_list = "\n".join(lines)
    return f"""Translate each Portuguese lemma to 1-3 Spanish dictionary-form glosses.

Lemmas:
{word_list}

Return JSON:
{{"translations": [{{"lemma": "...", "pos": "...", "spanish": ["...", "..."]}}, ...]}}

Include every lemma exactly once. Use dictionary forms only."""


def _extract_batch_results(
    items: list[tuple[str, str]],
    raw: str,
) -> dict[str, list[str]]:
    """Parse LLM JSON and extract valid lemma → gloss mappings."""
    if not raw:
        return {}

    data = json.loads(raw)
    batch = TranslationBatchResponse.from_json(data)
    results: dict[str, list[str]] = {}

    for entry in batch.translations:
        key = entry.lemma.lower()
        glosses: list[str] = []
        for gloss in entry.spanish:
            cleaned = _validate_spanish_gloss(gloss)
            if cleaned and cleaned not in glosses:
                glosses.append(cleaned)
        if glosses:
            results[key] = glosses

    return results


def _cache_lemma_results(
    results: dict[str, list[str]],
    items: list[tuple[str, str]],
    use_cache: bool,
) -> None:
    if not use_cache:
        return
    for lemma, pos in items:
        key = lemma.lower()
        if key in results:
            set_cached(
                "translation",
                {"spanish": results[key]},
                *_lemma_cache_key(lemma, pos),
            )


def _call_translation_api(
    items: list[tuple[str, str]],
    last_error: str = "",
) -> str:
    client = get_client()
    prompt = _build_batch_prompt(items)
    messages = [{"role": "system", "content": TRANSLATE_SYSTEM_PROMPT}]
    if last_error:
        messages.append(
            {
                "role": "user",
                "content": f"Previous attempt failed: {last_error}\n\n{prompt}",
            }
        )
    else:
        messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("Empty LLM response")
    return raw


def _translate_chunk_once(
    items: list[tuple[str, str]],
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """Translate a chunk, returning partial results without raising on misses."""
    if not items:
        return {}

    cache_key = _batch_cache_key(items)
    if use_cache:
        cached = get_cached("translation_batch", *cache_key)
        if cached and "results" in cached:
            expected = {lemma.lower() for lemma, _ in items}
            if expected <= set(cached["results"].keys()):
                return cached["results"]

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _call_translation_api(items, last_error)
            results = _extract_batch_results(items, raw)
            if use_cache and results:
                expected = {lemma.lower() for lemma, _ in items}
                if expected <= set(results.keys()):
                    set_cached("translation_batch", {"results": results}, *cache_key)
                _cache_lemma_results(results, items, use_cache=True)
            return results
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            last_error = str(exc)
            logger.warning(
                "Translation chunk attempt %d/%d failed (%d lemmas): %s",
                attempt,
                MAX_RETRIES,
                len(items),
                last_error,
            )
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)

    return {}


def translate_batch(
    items: list[tuple[str, str]],
    use_cache: bool = True,
) -> dict[str, list[str]]:
    """Translate a batch, saving partial results and retrying misses in smaller chunks."""
    if not items:
        return {}

    results: dict[str, list[str]] = {}
    remaining = list(items)
    chunk_sizes = [len(items), 10, 1]

    for chunk_size in chunk_sizes:
        if not remaining:
            break
        if chunk_size > len(remaining) and chunk_size != 1:
            continue

        next_remaining: list[tuple[str, str]] = []
        step = chunk_size if chunk_size != 1 else 1
        for i in range(0, len(remaining), step):
            chunk = remaining[i : i + step]
            partial = _translate_chunk_once(chunk, use_cache=use_cache)
            results.update(partial)
            for lemma, pos in chunk:
                key = lemma.lower()
                if key not in partial:
                    next_remaining.append((lemma, pos))

        if len(next_remaining) < len(remaining):
            logger.info(
                "Partial batch progress: %d translated, %d remaining",
                len(items) - len(next_remaining),
                len(next_remaining),
            )
        remaining = next_remaining

    if remaining:
        missed = [lemma for lemma, _ in remaining[:8]]
        logger.warning(
            "Could not translate %d lemmas after fallbacks: %s%s",
            len(remaining),
            missed,
            "..." if len(remaining) > 8 else "",
        )

    return results


def run_translate(
    input_path: str | None = None,
    top_n: int | None = None,
    dry_run: bool = False,
    use_cache: bool = True,
    refresh_lemma: str | None = None,
    seed_dict: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    """Build or update pt_es_dict.json for top-N lemmas."""
    from src.seed_dict import get_seed_dict as _get_seed

    path = Path(input_path) if input_path else CLEANED_PARQUET
    df = pd.read_parquet(path)
    limit = top_n if top_n is not None else TRANSLATE_TOP_N
    work = df.head(limit)

    existing = load_pt_es_dict()
    seeds = seed_dict if seed_dict is not None else _get_seed()
    for k, v in seeds.items():
        existing.setdefault(k.lower(), v)

    if refresh_lemma:
        work = work[work["lemma"] == refresh_lemma]
        for lemma in work["lemma"]:
            existing.pop(lemma.lower(), None)

    from src.classify import is_excluded_label, load_clean_labels, load_clean_overrides
    from src.classify import apply_overrides_to_labels

    clean_labels = apply_overrides_to_labels(
        load_clean_labels(),
        load_clean_overrides(),
    )
    skipped_exclude = 0

    to_translate: list[tuple[str, str]] = []
    for _, row in work.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        key = lemma.lower()
        if is_excluded_label(clean_labels.get(key)):
            skipped_exclude += 1
            continue
        if key in existing and not refresh_lemma:
            continue
        if use_cache:
            cached = get_cached("translation", *_lemma_cache_key(lemma, pos))
            if cached and cached.get("spanish"):
                existing[key] = cached["spanish"]
                continue
        to_translate.append((lemma, pos))

    logger.info(
        "Translate: %d lemmas in scope, %d excluded by classifier, %d already cached, %d to fetch",
        len(work),
        skipped_exclude,
        len(work) - skipped_exclude - len(to_translate),
        len(to_translate),
    )

    if dry_run:
        for lemma, pos in to_translate[:30]:
            logger.info("  would translate: %s (%s)", lemma, pos)
        if len(to_translate) > 30:
            logger.info("  ... and %d more", len(to_translate) - 30)
        return existing

    if to_translate:
        for i in tqdm(
            range(0, len(to_translate), TRANSLATE_BATCH_SIZE),
            desc="Translating PT→ES",
        ):
            batch = to_translate[i : i + TRANSLATE_BATCH_SIZE]
            results = translate_batch(batch, use_cache=use_cache)
            if results:
                existing.update(results)
                save_pt_es_dict(existing)
            else:
                logger.error("Batch at index %d returned no translations", i)

    save_pt_es_dict(existing)
    coverage = sum(1 for _, row in work.iterrows() if row["lemma"].lower() in existing)
    logger.info(
        "Translation coverage: %d/%d (%.1f%%)",
        coverage,
        len(work),
        100.0 * coverage / len(work) if len(work) else 0,
    )
    return existing
