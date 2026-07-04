"""Step 3: Rank lemmas by priority score."""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    DISSIMILARITY_EXPONENT,
    FREQUENCY_BLEND,
    RANKED_PARQUET,
    SCORED_PARQUET,
    SIMILARITY_FOCUS_THRESHOLD,
)

logger = logging.getLogger(__name__)


def dissimilarity_factor(
    similarity: float,
    threshold: float = SIMILARITY_FOCUS_THRESHOLD,
    exponent: float = DISSIMILARITY_EXPONENT,
) -> float:
    """Map similarity to a 0–1 weight; lower similarity yields higher weight."""
    headroom = max(threshold - similarity, 0.0)
    if threshold <= 0:
        return 0.0
    return (headroom / threshold) ** exponent


def is_study_candidate(row: pd.Series) -> bool:
    """Return True if a lemma should appear in the study deck."""
    if row.get("studyable") is False:
        return False
    if row.get("cognate_source") == "missing":
        return False
    if row.get("penalty", 0.0) > 0:
        return False
    return True


def compute_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute priority from frequency and translation dissimilarity."""
    ranked = df.copy()
    ranked["dissimilarity_factor"] = ranked["translation_similarity"].map(
        lambda sim: dissimilarity_factor(float(sim))
    )
    ranked["study_weight"] = FREQUENCY_BLEND + (1.0 - FREQUENCY_BLEND) * ranked[
        "dissimilarity_factor"
    ]
    ranked["priority_score"] = ranked["log10_freq_orto"] * ranked["study_weight"]
    ranked = ranked.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return ranked


def run_rank(
    input_path: str | None = None,
    output_path: str | None = None,
    top_n: int | None = None,
) -> pd.DataFrame:
    """Run ranking stage and optionally slice to top N."""
    in_path = Path(input_path) if input_path else SCORED_PARQUET
    out_path = Path(output_path) if output_path else RANKED_PARQUET

    df = pd.read_parquet(in_path)
    study_mask = df.apply(is_study_candidate, axis=1)
    excluded = int((~study_mask).sum())
    if excluded:
        logger.info("Excluding %d lemmas not suitable for study", excluded)
    df = df[study_mask]

    ranked = compute_priority_scores(df)

    if top_n is not None:
        ranked = ranked.head(top_n)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(out_path, index=False)
    logger.info("Wrote ranked data to %s (%d rows)", out_path, len(ranked))
    return ranked
