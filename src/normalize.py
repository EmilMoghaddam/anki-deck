"""String normalization for lemma comparison."""

import re
import unicodedata

from rapidfuzz import distance, fuzz

from src.config import SIMILARITY_FOCUS_THRESHOLD

# Common encoding corruption in LexPorBR (Romanian ă used instead of Portuguese ã)
CORRUPTION_MAP = {
    "ă": "ã",
    "Ă": "Ã",
    "Ť": "",
    "ť": "",
}

ACCENT_MAP = str.maketrans(
    "àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ",
    "aaaaaaeeeeiiiiooooouuuuyyncAAAAAAEEEEIIIIOOOOOUUUUYYNC",
)

LEMMA_RE = re.compile(r"^[a-zA-ZàáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ\-]+$")

# Map common PT orthography toward ES for cognate comparison (longer patterns first).
PT_ES_ORTHOGRAPHY_PATTERNS: list[tuple[str, str]] = [
    (r"ção$", "cion"),
    (r"ções$", "ciones"),
    (r"ães$", "anes"),
    (r"ões$", "ones"),
    (r"ão", "an"),
    (r"ã", "a"),
    (r"õ", "o"),
    (r"ç", "c"),
    (r"lh", "ll"),
    (r"nh", "n"),
    (r"ável$", "able"),
    (r"ível$", "ible"),
]


def repair_corruption(text: str) -> str:
    """Fix known encoding artifacts in LexPorBR."""
    for bad, good in CORRUPTION_MAP.items():
        text = text.replace(bad, good)
    return text


def normalize_unicode(text: str) -> str:
    """Apply NFKC normalization and corruption repair."""
    text = unicodedata.normalize("NFKC", text)
    return repair_corruption(text)


def strip_accents(text: str) -> str:
    """Remove diacritics for cross-language comparison."""
    text = normalize_unicode(text)
    return text.translate(ACCENT_MAP).lower()


def normalize_for_match(text: str) -> str:
    """Normalize lemma for cognate similarity comparison."""
    text = strip_accents(text)
    text = text.replace("-", "").replace("'", "")
    return text


def pt_to_es_orthography(text: str) -> str:
    """Approximate Spanish spelling from a PT lemma for cognate comparison."""
    result = normalize_for_match(text)
    for pattern, replacement in PT_ES_ORTHOGRAPHY_PATTERNS:
        result = re.sub(pattern, replacement, result)
    return result


def _common_prefix_len(a: str, b: str) -> int:
    count = 0
    for left, right in zip(a, b, strict=False):
        if left != right:
            break
        count += 1
    return count


def _clean_gloss(gloss: str) -> str:
    return gloss.split(",")[0].split(";")[0].strip()


def cognate_similarity(
    pt_lemma: str,
    es_gloss: str,
    *,
    threshold: float = SIMILARITY_FOCUS_THRESHOLD,
) -> float:
    """Estimate orthographic cognate similarity between a PT lemma and Spanish gloss.

  Uses accent-stripped comparison, PT→ES orthographic normalization, and guards
  against suffix-only false positives (e.g. chegar/llegar).
    """
    gloss = _clean_gloss(es_gloss)
    raw_a = normalize_for_match(pt_lemma)
    raw_b = normalize_for_match(gloss)
    ortho_a = pt_to_es_orthography(pt_lemma)
    ortho_b = raw_b

    if not raw_a or not raw_b:
        return 0.0

    ratio = max(
        float(fuzz.ratio(raw_a, raw_b)),
        float(fuzz.ratio(ortho_a, ortho_b)),
    )
    partial = float(fuzz.partial_ratio(ortho_a, ortho_b))
    prefix = _common_prefix_len(ortho_a, ortho_b)

    # Full embedding: onde/donde, tentar/intentar, apresentar/presentar.
    if partial >= 99.5:
        return max(ratio, partial)

    # Suffix overlap without shared onset (chegar/llegar) — do not inflate below threshold.
    if partial >= 80 and prefix == 0 and ratio < threshold:
        return ratio

    # Shared onset with vowel shifts: hoje/hoy, novo/nuevo, também/también.
    if ratio < threshold and prefix >= 1:
        jw = distance.JaroWinkler.similarity(ortho_a, ortho_b) * 100
        max_len = max(len(ortho_a), len(ortho_b))
        len_ratio = min(len(ortho_a), len(ortho_b)) / max_len if max_len else 1.0
        if len_ratio >= 0.65:
            return max(ratio, jw)

    return ratio


def is_valid_lemma(text: str) -> bool:
    """Check if text is a valid Portuguese lemma string."""
    if not text or len(text) < 2:
        return False
    if any(c.isdigit() for c in text):
        return False
    return bool(LEMMA_RE.match(text))
