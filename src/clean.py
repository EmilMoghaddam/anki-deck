"""Step 1: Clean LexPorBR frequency list."""

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    CLEANED_PARQUET,
    CONTENT_POS,
    INPUT_FILE,
    MIN_LEMMA_LENGTH,
    OUTPUT_DIR,
)
from src.clean_heuristics import exclusion_reason
from src.normalize import is_valid_lemma, normalize_unicode

logger = logging.getLogger(__name__)


def clean_frequency_list(input_path: str | None = None) -> pd.DataFrame:
    """Clean raw LexPorBR TSV and return content-word lemmas."""
    path = input_path or str(INPUT_FILE)
    logger.info("Loading frequency list from %s", path)

    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    logger.info("Loaded %d raw rows", len(df))

    # Normalize European decimal format in numeric columns
    for col in ("log10_freq_orto", "freq_orto/M", "zipf_escala"):
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."),
                errors="coerce",
            )
    if "freq_orto" in df.columns:
        df["freq_orto"] = pd.to_numeric(df["freq_orto"], errors="coerce")

    df = df.rename(
        columns={
            "ortografia": "lemma",
            "cat_gram": "pos",
        }
    )

    # Content words only
    df = df[df["pos"].isin(CONTENT_POS)]
    logger.info("After content-word filter: %d rows", len(df))

    # Normalize lemma text
    df["lemma"] = df["lemma"].astype(str).map(normalize_unicode)

    # Valid lexical form
    df = df[df["lemma"].map(is_valid_lemma)]
    df = df[df["lemma"].str.len() >= MIN_LEMMA_LENGTH]
    logger.info("After lexical validation: %d rows", len(df))

    # Heuristic noise filter (abbreviations, roman numerals, ...)
    reasons = df["lemma"].map(exclusion_reason)
    excluded = reasons.notna()
    if excluded.any():
        by_reason = reasons[excluded].value_counts()
        logger.info(
            "Heuristic exclusions: %d rows (%s)",
            int(excluded.sum()),
            ", ".join(f"{reason}={count}" for reason, count in by_reason.items()),
        )
        df = df[~excluded]
    logger.info("After heuristic filter: %d rows", len(df))

    # Deduplicate — keep highest frequency
    df = df.sort_values("freq_orto", ascending=False)
    df = df.drop_duplicates(subset="lemma", keep="first")

    # Assign frequency rank
    df = df.sort_values("freq_orto", ascending=False).reset_index(drop=True)
    df["freq_rank"] = df.index + 1

    result = df[["lemma", "pos", "freq_orto", "log10_freq_orto", "freq_rank"]].copy()
    logger.info("Final cleaned lemmas: %d", len(result))
    return result


def save_cleaned(df: pd.DataFrame, output_path: str | None = None) -> Path:
    """Write cleaned dataframe to parquet."""
    out = Path(output_path) if output_path else CLEANED_PARQUET
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Wrote cleaned data to %s", out)
    return out


def run_clean(input_path: str | None = None, output_path: str | None = None) -> pd.DataFrame:
    """Run cleaning stage and persist output."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = clean_frequency_list(input_path)
    save_cleaned(df, output_path)
    return df
