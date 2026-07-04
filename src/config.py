"""Pipeline configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
REFERENCE_DIR = DATA_DIR / "reference"
CACHE_DIR = DATA_DIR / "cache"
TTS_CACHE_DIR = CACHE_DIR / "tts"
OUTPUT_DIR = DATA_DIR / "output"

INPUT_FILE = INPUT_DIR / "lexporbr_alfa_lemas_txt.txt"
PT_ES_DICT_FILE = REFERENCE_DIR / "pt_es_dict.json"
COGNATE_OVERRIDES_FILE = REFERENCE_DIR / "cognate_overrides.csv"

CLEANED_PARQUET = OUTPUT_DIR / "cleaned.parquet"
SCORED_PARQUET = OUTPUT_DIR / "scored.parquet"
RANKED_PARQUET = OUTPUT_DIR / "ranked.parquet"
CARDS_JSONL = OUTPUT_DIR / "cards.jsonl"
ANKI_CSV = OUTPUT_DIR / "anki_deck.csv"
ANKI_APKG = OUTPUT_DIR / "pt_br_vocab.apkg"

# Content-word POS categories from LexPorBR
CONTENT_POS = frozenset({"nom", "ver", "adj", "adv"})

# Portuguese lemma pattern (letters, accents, hyphen)
LEMMA_PATTERN = r"^[a-zA-ZàáâãéêíóôõúçÀÁÂÃÉÊÍÓÔÕÚÇ\-]+$"
MIN_LEMMA_LENGTH = 2
MIN_LEMMA_LENGTH_FOR_COGNATE = 3

# Words with translation similarity >= this are excluded from the study deck.
SIMILARITY_FOCUS_THRESHOLD = 75

# How aggressively dissimilarity boosts rank below the threshold.
# study_weight = FREQUENCY_BLEND + (1 - FREQUENCY_BLEND) * dissimilarity_factor
# priority = log10_freq * study_weight
# dissimilarity_factor = ((threshold - similarity) / threshold) ** exponent
DISSIMILARITY_EXPONENT = 1.5

# Baseline share of frequency kept even for somewhat similar words (0–1).
FREQUENCY_BLEND = 0.35

# Legacy graduated tiers (unused while focus threshold applies flat penalty).
COGNATE_PENALTY_TIERS: list[tuple[int, float]] = [
    (100, 0.85),
    (95, 0.60),
    (90, 0.35),
    (85, 0.15),
    (80, 0.05),
]

# PT→ES translation for cognate filtering
TRANSLATE_TOP_N = 25_000
TRANSLATE_BATCH_SIZE = 50
TRANSLATE_PROMPT_VERSION = "v1"

# LLM studyability classification
CLEAN_LABELS_FILE = REFERENCE_DIR / "clean_labels.json"
CLEAN_OVERRIDES_FILE = REFERENCE_DIR / "clean_overrides.csv"
# Top 10k covers the study deck (max freq_rank in deck ~10k); 25k was too slow for marginal gain.
CLASSIFY_TOP_N = 10_000
CLASSIFY_BATCH_SIZE = 20
CLASSIFY_FALLBACK_CHUNK_SIZES = (5, 1)
CLASSIFY_PROMPT_VERSION = "v2"
CLASSIFY_TEMPERATURE = 0.1

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-chat"  # translate + classify
CARD_GENERATION_MODEL = "anthropic/claude-sonnet-4.6"
OPENROUTER_SITE_URL = "https://github.com/anki-deck"
OPENROUTER_APP_NAME = "pt-br-anki-deck"
TEMPERATURE = 0.3
MAX_CONCURRENT = 5
MAX_RETRIES = 3
PROMPT_VERSION = "v3"
CARD_EXAMPLE_COUNT = 5

# Edge TTS (Brazilian Portuguese) for Anki .apkg export
TTS_VOICE = "pt-BR-FranciscaNeural"
TTS_RATE = "+0%"
TTS_CONCURRENCY = 8

# Default deck size
DEFAULT_TOP_N = 10_000
