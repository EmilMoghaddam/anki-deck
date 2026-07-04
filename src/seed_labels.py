"""Seed studyability labels for bootstrap and smoke tests."""

from typing import Literal

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

SEED_LABELS: dict[str, dict[str, str]] = {
    "shopping": {"verdict": "exclude", "reason": "foreign_loanword", "pos": "nom"},
    "black": {"verdict": "exclude", "reason": "foreign_loanword", "pos": "adj"},
    "boom": {"verdict": "exclude", "reason": "foreign_loanword", "pos": "nom"},
    "km": {"verdict": "exclude", "reason": "abbreviation", "pos": "nom"},
    "tv": {"verdict": "exclude", "reason": "abbreviation", "pos": "nom"},
    "ontem": {"verdict": "keep", "reason": "valid", "pos": "adv"},
    "dizer": {"verdict": "keep", "reason": "valid", "pos": "ver"},
    "fazer": {"verdict": "keep", "reason": "valid", "pos": "ver"},
    "ficar": {"verdict": "keep", "reason": "valid", "pos": "ver"},
    "jeito": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "saudade": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "jornal": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "lixo": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "quarta-feira": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "prefeito": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "greve": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "hospital": {"verdict": "keep", "reason": "valid", "pos": "nom"},
    "chance": {"verdict": "exclude", "reason": "foreign_loanword", "pos": "nom"},
    "time": {"verdict": "exclude", "reason": "foreign_loanword", "pos": "nom"},
    "top": {"verdict": "exclude", "reason": "foreign_loanword", "pos": "nom"},
}


def get_seed_labels() -> dict[str, dict[str, str]]:
    """Return normalized seed labels keyed by lowercase lemma."""
    return {k.lower(): v for k, v in SEED_LABELS.items()}
