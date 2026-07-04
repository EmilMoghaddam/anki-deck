"""Heuristic filters for corpus noise in LexPorBR lemmas."""

import re

# Roman numerals tagged as lemmas in news corpora (ii, iii, xix, ...).
ROMAN_NUMERAL_RE = re.compile(r"^[ivxlc]+$", re.IGNORECASE)

# Corpus abbreviations, units, and shorthand — not useful for reading practice.
EXCLUDED_ABBREVIATIONS = frozenset(
    {
        "ag",
        "ai",
        "al",
        "ar",
        "av",
        "bi",
        "br",
        "cc",
        "cf",
        "cic",
        "cm",
        "cr",
        "cv",
        "db",
        "dj",
        "dp",
        "du",
        "ed",
        "en",
        "ex",
        "fm",
        "fr",
        "ha",
        "hp",
        "kg",
        "km",
        "md",
        "mg",
        "ml",
        "mm",
        "np",
        "op",
        "pc",
        "ph",
        "pi",
        "pp",
        "qi",
        "rg",
        "rh",
        "sr",
        "st",
        "tb",
        "th",
        "tv",
        "un",
        "up",
    }
)


def exclusion_reason(lemma: str) -> str | None:
    """Return exclusion reason, or None if the lemma should be kept."""
    key = lemma.lower()
    if key in EXCLUDED_ABBREVIATIONS:
        return "abbreviation"
    if ROMAN_NUMERAL_RE.fullmatch(key) and len(key) >= 2:
        return "roman_numeral"
    return None


def should_exclude_lemma(lemma: str) -> bool:
    """Return True if a lemma should be dropped during cleaning."""
    return exclusion_reason(lemma) is not None
