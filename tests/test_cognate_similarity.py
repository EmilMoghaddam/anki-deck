"""Tests for PT–ES cognate similarity scoring."""

import pytest

from src.config import SIMILARITY_FOCUS_THRESHOLD
from src.normalize import cognate_similarity


@pytest.mark.parametrize(
    ("pt", "es", "should_exclude"),
    [
        ("chegar", "llegar", False),
        ("chamar", "llamar", False),
        ("ficar", "quedar", False),
        ("ontem", "ayer", False),
        ("hoje", "hoy", True),
        ("também", "también", True),
        ("governo", "gobierno", True),
        ("receber", "recibir", True),
        ("onde", "donde", True),
        ("falar", "hablar", False),  # score ~73, below 75 threshold
        ("quando", "cuándo", True),
        ("milhão", "millón", True),
        ("novo", "nuevo", True),
    ],
)
def test_cognate_similarity_cases(pt: str, es: str, should_exclude: bool) -> None:
    score = cognate_similarity(pt, es)
    assert (score >= SIMILARITY_FOCUS_THRESHOLD) == should_exclude
