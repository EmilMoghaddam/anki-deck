"""Pydantic models for pipeline data and LLM output."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Example(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pt: str
    en: str


class VocabCard(BaseModel):
    model_config = ConfigDict(extra="ignore")

    lemma: str
    part_of_speech: str
    english_translations: list[str] = Field(min_length=1, max_length=4)
    spanish_translations: list[str] = Field(min_length=1, max_length=4)
    examples: list[Example] = Field(min_length=5, max_length=5)

    @field_validator("english_translations", "spanish_translations")
    @classmethod
    def validate_translation_length(cls, v: list[str]) -> list[str]:
        for item in v:
            words = item.strip().split()
            if len(words) > 3:
                raise ValueError(f"Translation too long (max 3 words): {item!r}")
        return v


class LemmaRecord(BaseModel):
    lemma: str
    pos: str
    freq_orto: float
    log10_freq_orto: float
    freq_rank: int
    es_translation: str = ""
    translation_similarity: float = 0.0
    penalty: float = 0.0
    cognate_source: str = "missing"
    priority_score: float = 0.0
