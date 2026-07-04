"""Step 2: Translation-first cognate filtering."""

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.config import (
    CLEANED_PARQUET,
    COGNATE_OVERRIDES_FILE,
    MIN_LEMMA_LENGTH_FOR_COGNATE,
    PT_ES_DICT_FILE,
    SCORED_PARQUET,
    SIMILARITY_FOCUS_THRESHOLD,
    CLASSIFY_TOP_N,
)
from src.normalize import cognate_similarity

logger = logging.getLogger(__name__)


def penalty_from_similarity(similarity: float) -> float:
    """Map similarity score to cognate penalty.

    Below SIMILARITY_FOCUS_THRESHOLD: no penalty (eligible for study deck).
    At or above: penalty 1.0 (excluded from study deck at rank time).
    """
    if similarity < SIMILARITY_FOCUS_THRESHOLD:
        return 0.0
    return 1.0


def load_pt_es_dict(path: Path | None = None) -> dict[str, list[str]]:
    """Load PT→ES translation dictionary."""
    ref_path = path or PT_ES_DICT_FILE
    if not ref_path.exists():
        logger.warning("PT-ES dict not found at %s", ref_path)
        return {}
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    logger.info("Loaded PT-ES dict with %d entries", len(data))
    return {k.lower(): v for k, v in data.items()}


def load_cognate_overrides(path: Path | None = None) -> dict[str, str]:
    """Load manual cognate overrides (lemma -> keep|deprioritize)."""
    ref_path = path or COGNATE_OVERRIDES_FILE
    if not ref_path.exists():
        return {}
    df = pd.read_csv(ref_path)
    if df.empty or "lemma" not in df.columns:
        return {}
    overrides = {}
    for _, row in df.iterrows():
        lemma = str(row["lemma"]).strip()
        action = str(row.get("action", "")).strip().lower()
        if lemma and action in ("keep", "deprioritize"):
            overrides[lemma] = action
    logger.info("Loaded %d cognate overrides", len(overrides))
    return overrides


def cognate_penalty_from_translations(
    lemma: str,
    translations: list[str],
) -> tuple[float, str, float, str]:
    """Score cognate penalty from PT lemma vs Spanish gloss similarity."""
    if len(lemma) < MIN_LEMMA_LENGTH_FOR_COGNATE or not translations:
        return 0.0, "", 0.0, "missing"

    best_gloss = ""
    best_sim = 0.0

    for gloss in translations:
        sim = cognate_similarity(lemma, gloss)
        if sim > best_sim:
            best_gloss = gloss.split(",")[0].split(";")[0].strip()
            best_sim = sim

    penalty = penalty_from_similarity(best_sim)
    return penalty, best_gloss, best_sim, "translation"


def _studyability_columns(df: pd.DataFrame) -> tuple[list[bool], list[str]]:
    """Build studyable and exclude_reason columns from clean labels."""
    from src.classify import (
        apply_overrides_to_labels,
        is_excluded_label,
        load_clean_labels,
        load_clean_overrides,
    )

    labels = apply_overrides_to_labels(load_clean_labels(), load_clean_overrides())
    studyable: list[bool] = []
    exclude_reasons: list[str] = []

    for _, row in df.iterrows():
        freq_rank = int(row["freq_rank"])
        key = row["lemma"].lower()
        if freq_rank > CLASSIFY_TOP_N:
            studyable.append(True)
            exclude_reasons.append("")
            continue
        label = labels.get(key)
        if is_excluded_label(label):
            studyable.append(False)
            exclude_reasons.append(label.get("reason", "other") if label else "")
        else:
            studyable.append(True)
            exclude_reasons.append("")

    return studyable, exclude_reasons


def apply_cognate_filter(
    df: pd.DataFrame,
    pt_es_dict: dict[str, list[str]] | None = None,
    overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Apply translation-first cognate filtering."""
    pt_es_dict = pt_es_dict if pt_es_dict is not None else load_pt_es_dict()
    overrides = overrides if overrides is not None else load_cognate_overrides()

    penalties = []
    es_translations = []
    similarities = []
    sources = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cognate filter"):
        lemma = row["lemma"]
        translations = pt_es_dict.get(lemma.lower(), [])
        penalty, best_gloss, sim, source = cognate_penalty_from_translations(
            lemma, translations
        )

        if lemma in overrides:
            if overrides[lemma] == "deprioritize":
                penalty = 0.85
            else:
                penalty = 0.0
            source = "override"

        penalties.append(penalty)
        es_translations.append(best_gloss)
        similarities.append(sim)
        sources.append(source)

    scored = df.copy().reset_index(drop=True)
    scored["es_translation"] = es_translations
    scored["translation_similarity"] = similarities
    scored["penalty"] = penalties
    scored["cognate_source"] = sources

    studyable, exclude_reasons = _studyability_columns(scored)
    scored["studyable"] = studyable
    scored["exclude_reason"] = exclude_reasons

    with_trans = sum(1 for s in sources if s == "translation")
    with_penalty = (scored["penalty"] > 0).sum()
    logger.info(
        "Cognate filtering complete: %d with translations, %d with penalty > 0",
        with_trans,
        with_penalty,
    )
    return scored


def run_filter(
    input_path: str | None = None,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run cognate filtering stage."""
    in_path = Path(input_path) if input_path else CLEANED_PARQUET
    out_path = Path(output_path) if output_path else SCORED_PARQUET

    df = pd.read_parquet(in_path)
    scored = apply_cognate_filter(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out_path, index=False)
    logger.info("Wrote scored data to %s", out_path)
    return scored
