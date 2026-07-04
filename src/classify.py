"""LLM studyability classifier for cleaned lemmas."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from src.cache import get_cached, set_cached
from src.config import (
    CLASSIFY_BATCH_SIZE,
    CLASSIFY_FALLBACK_CHUNK_SIZES,
    CLASSIFY_PROMPT_VERSION,
    CLASSIFY_TEMPERATURE,
    CLASSIFY_TOP_N,
    CLEANED_PARQUET,
    CLEAN_LABELS_FILE,
    CLEAN_OVERRIDES_FILE,
    MAX_RETRIES,
    OPENROUTER_MODEL,
)
from src.openrouter_client import get_client

logger = logging.getLogger(__name__)

POS_LABELS = {
    "nom": "noun",
    "ver": "verb",
    "adj": "adjective",
    "adv": "adverb",
}

Verdict = Literal["keep", "exclude"]
Reason = Literal[
    "valid",
    "foreign_loanword",
    "proper_noun",
    "acronym",
    "abbreviation",
    "typo",
    "corpus_artifact",
    "other",
]

CLASSIFY_SYSTEM_PROMPT = """You are a lexicographer curating a Brazilian Portuguese vocabulary list.

A Spanish B2 speaker will use this list to learn words needed for reading Brazilian Portuguese news and media.
Output valid JSON only, no markdown.

For each lemma, decide whether it belongs in a study deck.

KEEP:
- Real Portuguese dictionary words, including slang common in Brazilian media
- Hyphen compounds that are standard vocabulary (quarta-feira, fim-de-semana)

EXCLUDE:
- English or other foreign loanwords a Spanish speaker already knows (shopping, black, boom, time)
- Proper nouns, acronyms, abbreviations, units
- Corpus typos, OCR errors, and non-words

When unsure: keep high-frequency common Portuguese words; exclude obvious non-Portuguese tokens."""


class ClassificationItem(BaseModel):
    lemma: str
    pos: str = ""
    verdict: Verdict
    reason: Reason


class ClassificationBatchResponse(BaseModel):
    classifications: list[ClassificationItem]

    @classmethod
    def from_json(cls, data: object) -> "ClassificationBatchResponse":
        if isinstance(data, list):
            return cls(
                classifications=[
                    ClassificationItem.model_validate(item) for item in data
                ]
            )
        if isinstance(data, dict):
            entries = data.get("classifications", data.get("results", []))
            if isinstance(entries, list):
                return cls(
                    classifications=[
                        ClassificationItem.model_validate(item) for item in entries
                    ]
                )
        raise ValueError("Expected JSON object with classifications array")


def _lemma_cache_key(lemma: str, pos: str, freq_rank: int) -> tuple[str, ...]:
    return (
        lemma.lower(),
        pos,
        str(freq_rank),
        CLASSIFY_PROMPT_VERSION,
        OPENROUTER_MODEL,
    )


def _batch_cache_key(items: list[tuple[str, str, int]]) -> tuple[str, ...]:
    payload = "|".join(f"{l}:{p}:{r}" for l, p, r in sorted(items))
    return (payload, CLASSIFY_PROMPT_VERSION, OPENROUTER_MODEL)


def load_clean_labels(path: Path | None = None) -> dict[str, dict[str, str]]:
    """Load lemma studyability labels from JSON."""
    ref_path = path or CLEAN_LABELS_FILE
    if not ref_path.exists():
        return {}
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    return {k.lower(): v for k, v in data.items()}


def save_clean_labels(
    data: dict[str, dict[str, str]],
    path: Path | None = None,
) -> Path:
    """Persist lemma studyability labels."""
    out = path or CLEAN_LABELS_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    sorted_data = dict(sorted(data.items()))
    out.write_text(
        json.dumps(sorted_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote clean labels with %d entries to %s", len(sorted_data), out)
    return out


def load_clean_overrides(path: Path | None = None) -> dict[str, str]:
    """Load manual keep/exclude overrides."""
    ref_path = path or CLEAN_OVERRIDES_FILE
    if not ref_path.exists():
        return {}
    df = pd.read_csv(ref_path)
    if df.empty or "lemma" not in df.columns:
        return {}
    overrides: dict[str, str] = {}
    for _, row in df.iterrows():
        lemma = str(row["lemma"]).strip()
        action = str(row.get("action", "")).strip().lower()
        if lemma and action in ("keep", "exclude"):
            overrides[lemma] = action
    logger.info("Loaded %d clean overrides", len(overrides))
    return overrides


def label_to_entry(
    verdict: str,
    reason: str,
    pos: str = "",
) -> dict[str, str]:
    """Build a normalized label entry."""
    return {"verdict": verdict, "reason": reason, "pos": pos}


def is_excluded_label(label: dict[str, str] | None) -> bool:
    """Return True if a label marks the lemma as excluded."""
    return bool(label and label.get("verdict") == "exclude")


def studyable_from_label(label: dict[str, str] | None) -> bool:
    """Return True if a lemma should be studied (default keep when unknown)."""
    if not label:
        return True
    return label.get("verdict") != "exclude"


def apply_overrides_to_labels(
    labels: dict[str, dict[str, str]],
    overrides: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Apply manual overrides onto label dict (in place copy)."""
    merged = dict(labels)
    for lemma, action in overrides.items():
        key = lemma.lower()
        if action == "exclude":
            merged[key] = label_to_entry("exclude", "other", merged.get(key, {}).get("pos", ""))
        else:
            merged[key] = label_to_entry("keep", "valid", merged.get(key, {}).get("pos", ""))
    return merged


def _build_batch_prompt(items: list[tuple[str, str, int]]) -> str:
    lines = []
    for lemma, pos, freq_rank in items:
        pos_label = POS_LABELS.get(pos, pos)
        lines.append(f"- {lemma} ({pos_label}), rank {freq_rank}")
    word_list = "\n".join(lines)
    return f"""Classify each lemma (keep or exclude) for a Spanish B2 speaker learning to read Brazilian Portuguese.

{word_list}

Return JSON only:
{{"classifications": [{{"lemma": "...", "pos": "...", "verdict": "keep|exclude", "reason": "valid|foreign_loanword|proper_noun|acronym|abbreviation|typo|corpus_artifact|other"}}]}}

Every lemma exactly once."""


def _parse_json_response(raw: str) -> object:
    """Parse LLM output, tolerating markdown fences and surrounding text."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty LLM response")

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _extract_batch_results(raw: str) -> dict[str, dict[str, str]]:
    """Parse LLM JSON and extract valid lemma classifications."""
    data = _parse_json_response(raw)
    batch = ClassificationBatchResponse.from_json(data)
    results: dict[str, dict[str, str]] = {}

    for entry in batch.classifications:
        key = entry.lemma.lower()
        results[key] = label_to_entry(entry.verdict, entry.reason, entry.pos)

    return results


def _cache_lemma_results(
    results: dict[str, dict[str, str]],
    items: list[tuple[str, str, int]],
    use_cache: bool,
) -> None:
    if not use_cache:
        return
    for lemma, pos, freq_rank in items:
        key = lemma.lower()
        if key in results:
            set_cached(
                "classification",
                results[key],
                *_lemma_cache_key(lemma, pos, freq_rank),
            )


def _call_classification_api(
    items: list[tuple[str, str, int]],
    last_error: str = "",
) -> str:
    client = get_client()
    prompt = _build_batch_prompt(items)
    messages = [{"role": "system", "content": CLASSIFY_SYSTEM_PROMPT}]
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
        temperature=CLASSIFY_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    return raw


def _classify_chunk_once(
    items: list[tuple[str, str, int]],
    use_cache: bool = True,
) -> dict[str, dict[str, str]]:
    """Classify a chunk, returning partial results without raising on misses."""
    if not items:
        return {}

    cache_key = _batch_cache_key(items)
    if use_cache:
        cached = get_cached("classification_batch", *cache_key)
        if cached and "results" in cached:
            expected = {lemma.lower() for lemma, _, _ in items}
            if expected <= set(cached["results"].keys()):
                return cached["results"]

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _call_classification_api(items, last_error)
            results = _extract_batch_results(raw)
            if use_cache and results:
                _cache_lemma_results(results, items, use_cache=True)
                expected = {lemma.lower() for lemma, _, _ in items}
                if expected <= set(results.keys()):
                    set_cached(
                        "classification_batch",
                        {"results": results},
                        *cache_key,
                    )
            return results
        except (json.JSONDecodeError, ValueError, Exception) as exc:
            last_error = str(exc)
            logger.warning(
                "Classification chunk attempt %d/%d failed (%d lemmas): %s",
                attempt,
                MAX_RETRIES,
                len(items),
                last_error,
            )
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)

    return {}


def classify_batch(
    items: list[tuple[str, str, int]],
    use_cache: bool = True,
) -> dict[str, dict[str, str]]:
    """Classify a batch, saving partial results and retrying misses in smaller chunks."""
    if not items:
        return {}

    results: dict[str, dict[str, str]] = {}
    remaining = list(items)
    chunk_sizes = [len(items), *CLASSIFY_FALLBACK_CHUNK_SIZES]

    for chunk_size in chunk_sizes:
        if not remaining:
            break
        if chunk_size > len(remaining) and chunk_size != 1:
            continue

        next_remaining: list[tuple[str, str, int]] = []
        step = chunk_size if chunk_size != 1 else 1
        for i in range(0, len(remaining), step):
            chunk = remaining[i : i + step]
            partial = _classify_chunk_once(chunk, use_cache=use_cache)
            results.update(partial)
            for lemma, pos, freq_rank in chunk:
                key = lemma.lower()
                if key not in partial:
                    next_remaining.append((lemma, pos, freq_rank))

        if len(next_remaining) < len(remaining):
            logger.info(
                "Partial batch progress: %d classified, %d remaining",
                len(items) - len(next_remaining),
                len(next_remaining),
            )
        remaining = next_remaining

    if remaining:
        missed = [lemma for lemma, _, _ in remaining[:8]]
        logger.warning(
            "Could not classify %d lemmas after fallbacks (defaulting to keep): %s%s",
            len(remaining),
            missed,
            "..." if len(remaining) > 8 else "",
        )

    return results


def run_classify(
    input_path: str | None = None,
    top_n: int | None = None,
    dry_run: bool = False,
    use_cache: bool = True,
    refresh_lemma: str | None = None,
    seed_labels: dict[str, dict[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Build or update clean_labels.json for top-N lemmas."""
    from src.seed_labels import get_seed_labels as _get_seed

    path = Path(input_path) if input_path else CLEANED_PARQUET
    df = pd.read_parquet(path)
    limit = top_n if top_n is not None else CLASSIFY_TOP_N
    work = df.head(limit)

    existing = load_clean_labels()
    seeds = seed_labels if seed_labels is not None else _get_seed()
    for k, v in seeds.items():
        existing.setdefault(k.lower(), v)

    overrides = load_clean_overrides()
    existing = apply_overrides_to_labels(existing, overrides)

    if refresh_lemma:
        work = work[work["lemma"] == refresh_lemma]
        existing.pop(refresh_lemma.lower(), None)

    to_classify: list[tuple[str, str, int]] = []
    for _, row in work.iterrows():
        lemma = row["lemma"]
        pos = row["pos"]
        freq_rank = int(row["freq_rank"])
        key = lemma.lower()
        if key in existing and not refresh_lemma:
            continue
        if use_cache:
            cached = get_cached(
                "classification",
                *_lemma_cache_key(lemma, pos, freq_rank),
            )
            if cached and cached.get("verdict"):
                existing[key] = {
                    "verdict": cached["verdict"],
                    "reason": cached.get("reason", "other"),
                    "pos": cached.get("pos", pos),
                }
                continue
        to_classify.append((lemma, pos, freq_rank))

    logger.info(
        "Classify: %d lemmas in scope, %d already labeled, %d to fetch",
        len(work),
        len(work) - len(to_classify),
        len(to_classify),
    )

    if dry_run:
        for lemma, pos, freq_rank in to_classify[:30]:
            logger.info("  would classify: %s (%s) rank %d", lemma, pos, freq_rank)
        if len(to_classify) > 30:
            logger.info("  ... and %d more", len(to_classify) - 30)
        return existing

    if to_classify:
        for i in tqdm(
            range(0, len(to_classify), CLASSIFY_BATCH_SIZE),
            desc="Classifying lemmas",
        ):
            batch = to_classify[i : i + CLASSIFY_BATCH_SIZE]
            results = classify_batch(batch, use_cache=use_cache)
            if results:
                existing.update(results)
                existing = apply_overrides_to_labels(existing, overrides)
                save_clean_labels(existing)
            elif len(batch) > 1:
                logger.warning(
                    "Batch at index %d returned no classifications, retrying as singles",
                    i,
                )
                for item in batch:
                    single = classify_batch([item], use_cache=use_cache)
                    if single:
                        existing.update(single)
                        existing = apply_overrides_to_labels(existing, overrides)
                        save_clean_labels(existing)
            else:
                logger.error("Could not classify lemma: %s", batch[0][0])

    existing = apply_overrides_to_labels(existing, overrides)
    save_clean_labels(existing)

    labeled = sum(1 for _, row in work.iterrows() if row["lemma"].lower() in existing)
    excluded = sum(
        1
        for _, row in work.iterrows()
        if is_excluded_label(existing.get(row["lemma"].lower()))
    )
    logger.info(
        "Classification coverage: %d/%d labeled, %d excluded in scope",
        labeled,
        len(work),
        excluded,
    )
    return existing
