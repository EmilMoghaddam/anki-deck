"""Step 5: Export Anki CSV and .apkg deck."""

import csv
import json
import logging
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import genanki

from src.config import (
    ANKI_APKG,
    ANKI_AUDIO_MODE_DEFAULT,
    ANKI_CSV,
    ANKI_TTS_LANG,
    CARD_EXAMPLE_COUNT,
    CARDS_JSONL,
)
from src.tts import (
    audio_path_for_text,
    collect_card_audio_items,
    ensure_anki_compatible,
    ensure_audio,
    is_valid_audio,
    sound_field,
    strip_html,
)

logger = logging.getLogger(__name__)

# Stable IDs so re-exports update the same note type in Anki.
# v7: native device TTS (AnkiMobile-friendly). v6 edge MP3 kept via --edge-audio.
ANKI_MODEL_ID_NATIVE = 1_985_092_389
ANKI_MODEL_ID_EDGE = 1_985_092_388
ANKI_DECK_ID = 1_985_092_384
ANKI_DECK_NAME = "Brazilian Portuguese (ES speaker)"

NATIVE_FIELD_NAMES = [
    "Lemma",
    "POS",
    "English",
    "Spanish",
    *[f"Ex{i}_PT" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
    *[f"Ex{i}_TTS" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
    *[f"Ex{i}_EN" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
]

PLAIN_FIELD_NAMES = [
    "Lemma",
    "POS",
    "English",
    "Spanish",
    *[f"Ex{i}_PT" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
    *[f"Ex{i}_EN" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
]

EDGE_FIELD_NAMES = [
    "Lemma",
    "POS",
    "English",
    "Spanish",
    *[f"Ex{i}_PT" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
    *[f"Ex{i}_EN" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
    "LemmaAudio",
    *[f"Ex{i}_Audio" for i in range(1, CARD_EXAMPLE_COUNT + 1)],
]

ANKI_CSS = """
.card { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 18px; }
.lemma { font-size: 1.35em; font-weight: bold; margin-bottom: 0.6em; }
.pos { font-size: 0.75em; font-weight: normal; color: #666; }
.gloss { margin-bottom: 0.8em; }
.gloss .es { color: #555; font-size: 0.95em; }
.examples { text-align: left; line-height: 1.45; }
.example { margin: 0.7em 0; padding-bottom: 0.5em; border-bottom: 1px solid #eee; }
.example:last-child { border-bottom: none; }
.en { color: #444; font-style: italic; }
.tts { margin-top: 0.35em; }
"""


def _native_front(lang: str) -> str:
    return f"""
<div class="lemma">{{{{Lemma}}}} <span class="pos">({{{{POS}}}})</span></div>
<div class="tts">{{{{tts {lang}:Lemma}}}}</div>
""".strip()


def _native_back(lang: str) -> str:
    examples = "\n".join(
        f'<div class="example">{{{{Ex{i}_PT}}}}<br>'
        f'<div class="tts">{{{{tts {lang}:Ex{i}_TTS}}}}</div>'
        f'<span class="en">{{{{Ex{i}_EN}}}}</span></div>'
        for i in range(1, CARD_EXAMPLE_COUNT + 1)
    )
    return f"""
<div class="gloss">
<div><b>{{{{English}}}}</b></div>
<div class="es">{{{{Spanish}}}}</div>
</div>
<div class="examples">
{examples}
</div>
""".strip()


ANKI_FRONT_EDGE = """
<div class="lemma">{{Lemma}} <span class="pos">({{POS}})</span></div>
<div class="tts">{{LemmaAudio}}</div>
""".strip()

ANKI_BACK_EDGE = """
<div class="gloss">
<div><b>{{English}}</b></div>
<div class="es">{{Spanish}}</div>
</div>
<div class="examples">
<div class="example">{{Ex1_PT}}<br><div class="tts">{{Ex1_Audio}}</div><br><span class="en">{{Ex1_EN}}</span></div>
<div class="example">{{Ex2_PT}}<br><div class="tts">{{Ex2_Audio}}</div><br><span class="en">{{Ex2_EN}}</span></div>
<div class="example">{{Ex3_PT}}<br><div class="tts">{{Ex3_Audio}}</div><br><span class="en">{{Ex3_EN}}</span></div>
<div class="example">{{Ex4_PT}}<br><div class="tts">{{Ex4_Audio}}</div><br><span class="en">{{Ex4_EN}}</span></div>
<div class="example">{{Ex5_PT}}<br><div class="tts">{{Ex5_Audio}}</div><br><span class="en">{{Ex5_EN}}</span></div>
</div>
""".strip()


ANKI_FRONT_NONE = """
<div class="lemma">{{Lemma}} <span class="pos">({{POS}})</span></div>
""".strip()

ANKI_BACK_NONE = """
<div class="gloss">
<div><b>{{English}}</b></div>
<div class="es">{{Spanish}}</div>
</div>
<div class="examples">
<div class="example">{{Ex1_PT}}<br><span class="en">{{Ex1_EN}}</span></div>
<div class="example">{{Ex2_PT}}<br><span class="en">{{Ex2_EN}}</span></div>
<div class="example">{{Ex3_PT}}<br><span class="en">{{Ex3_EN}}</span></div>
<div class="example">{{Ex4_PT}}<br><span class="en">{{Ex4_EN}}</span></div>
<div class="example">{{Ex5_PT}}<br><span class="en">{{Ex5_EN}}</span></div>
</div>
""".strip()

CSV_COLUMNS = [
    "Lemma",
    "PartOfSpeech",
    "FrequencyRank",
    "EnglishTranslations",
    "SpanishTranslations",
    "Sentence1_PT",
    "Sentence1_EN",
    "Sentence2_PT",
    "Sentence2_EN",
    "Sentence3_PT",
    "Sentence3_EN",
    "Sentence4_PT",
    "Sentence4_EN",
    "Sentence5_PT",
    "Sentence5_EN",
    "PriorityScore",
]

POS_LABELS = {
    "nom": "noun",
    "ver": "verb",
    "adj": "adjective",
    "adv": "adverb",
}


def load_cards_jsonl(path: str | None = None) -> list[dict]:
    """Load all card records from jsonl."""
    jsonl_path = Path(path) if path else CARDS_JSONL
    if not jsonl_path.exists():
        raise FileNotFoundError(f"No cards file found: {jsonl_path}. Run generate first.")

    cards = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cards.append(json.loads(line))
    logger.info("Loaded %d cards from %s", len(cards), jsonl_path)
    return cards


def card_to_row(card: dict) -> dict:
    """Convert a card record to a CSV row dict."""
    pos_code = card.get("pos_code", "")
    pos_label = card.get("part_of_speech") or POS_LABELS.get(pos_code, pos_code)

    row = {
        "Lemma": card["lemma"],
        "PartOfSpeech": pos_label,
        "FrequencyRank": card.get("freq_rank", ""),
        "EnglishTranslations": ";".join(card.get("english_translations", [])),
        "SpanishTranslations": ";".join(card.get("spanish_translations", [])),
        "PriorityScore": card.get("priority_score", ""),
    }

    examples = card.get("examples", [])
    for i in range(CARD_EXAMPLE_COUNT):
        n = i + 1
        if i < len(examples):
            ex = examples[i]
            row[f"Sentence{n}_PT"] = ex.get("pt", "")
            row[f"Sentence{n}_EN"] = ex.get("en", "")
        else:
            row[f"Sentence{n}_PT"] = ""
            row[f"Sentence{n}_EN"] = ""

    return row


def card_to_native_fields(card: dict) -> list[str]:
    """Plain-text TTS fields + HTML display fields for device speech."""
    row = card_to_row(card)
    examples = _all_examples(card)
    return [
        card["lemma"],
        row["PartOfSpeech"],
        row["EnglishTranslations"].replace(";", ", "),
        row["SpanishTranslations"].replace(";", ", "),
        *[ex.get("pt", "") for ex in examples],
        *[strip_html(ex.get("pt", "")) for ex in examples],
        *[ex.get("en", "") for ex in examples],
    ]


def card_to_edge_fields(card: dict) -> list[str]:
    """Separate [sound:…] fields (no HTML in audio fields) for bundled MP3."""
    row = card_to_row(card)
    examples = _all_examples(card)
    lemma_path = audio_path_for_text(card["lemma"])
    lemma_audio = sound_field(lemma_path.name) if is_valid_audio(lemma_path) else ""
    audio_fields = []
    for ex in examples:
        path = audio_path_for_text(ex.get("pt", ""))
        audio_fields.append(sound_field(path.name) if is_valid_audio(path) else "")
    return [
        card["lemma"],
        row["PartOfSpeech"],
        row["EnglishTranslations"].replace(";", ", "),
        row["SpanishTranslations"].replace(";", ", "),
        *[ex.get("pt", "") for ex in examples],
        *[ex.get("en", "") for ex in examples],
        lemma_audio,
        *audio_fields,
    ]


def card_to_plain_fields(card: dict) -> list[str]:
    """Text-only fields (no audio)."""
    row = card_to_row(card)
    examples = _all_examples(card)
    return [
        card["lemma"],
        row["PartOfSpeech"],
        row["EnglishTranslations"].replace(";", ", "),
        row["SpanishTranslations"].replace(";", ", "),
        *[ex.get("pt", "") for ex in examples],
        *[ex.get("en", "") for ex in examples],
    ]


def build_anki_model(audio_mode: str) -> genanki.Model:
    """Return note type for native TTS, bundled edge audio, or text-only."""
    if audio_mode == "native":
        return genanki.Model(
            ANKI_MODEL_ID_NATIVE,
            "PT-BR Vocab (ES speaker)",
            fields=[{"name": name} for name in NATIVE_FIELD_NAMES],
            templates=[
                {
                    "name": "PT → EN",
                    "qfmt": _native_front(ANKI_TTS_LANG),
                    "afmt": _native_back(ANKI_TTS_LANG),
                }
            ],
            css=ANKI_CSS,
        )
    if audio_mode == "edge":
        return genanki.Model(
            ANKI_MODEL_ID_EDGE,
            "PT-BR Vocab (ES speaker)",
            fields=[{"name": name} for name in EDGE_FIELD_NAMES],
            templates=[
                {
                    "name": "PT → EN",
                    "qfmt": ANKI_FRONT_EDGE,
                    "afmt": ANKI_BACK_EDGE,
                }
            ],
            css=ANKI_CSS,
        )
    return genanki.Model(
        ANKI_MODEL_ID_NATIVE,
        "PT-BR Vocab (ES speaker)",
        fields=[{"name": name} for name in PLAIN_FIELD_NAMES],
        templates=[
            {
                "name": "PT → EN",
                "qfmt": ANKI_FRONT_NONE,
                "afmt": ANKI_BACK_NONE,
            }
        ],
        css=ANKI_CSS,
    )


def _all_examples(card: dict) -> list[dict]:
    """Return up to CARD_EXAMPLE_COUNT examples in generation order."""
    examples = list(card.get("examples", []))
    while len(examples) < CARD_EXAMPLE_COUNT:
        examples.append({"pt": "", "en": ""})
    return examples[:CARD_EXAMPLE_COUNT]


def disable_apkg_autoplay(apkg_path: Path) -> None:
    """Turn off automatic audio playback in the deck options embedded in .apkg."""
    tmp = apkg_path.with_suffix(".tmp.apkg")
    with zipfile.ZipFile(apkg_path, "r") as zin:
        with tempfile.TemporaryDirectory() as td:
            col_path = Path(td) / "collection.anki2"
            zin.extract("collection.anki2", td)
            conn = sqlite3.connect(col_path)
            dconf = json.loads(conn.execute("SELECT dconf FROM col").fetchone()[0])
            for cfg in dconf.values():
                cfg["autoplay"] = False
                cfg["replayq"] = False
            conn.execute("UPDATE col SET dconf = ?", (json.dumps(dconf),))
            conn.commit()
            conn.close()
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    if item.filename == "collection.anki2":
                        zout.write(col_path, "collection.anki2")
                    else:
                        zout.writestr(item, zin.read(item.filename))
    tmp.replace(apkg_path)


def export_anki_apkg(
    cards: list[dict] | None = None,
    output_path: str | None = None,
    deck_name: str = ANKI_DECK_NAME,
    audio_mode: str = "native",
) -> Path:
    """Write a ready-to-import .apkg file with note type and deck."""
    if cards is None:
        cards = load_cards_jsonl()

    out = Path(output_path) if output_path else ANKI_APKG
    out.parent.mkdir(parents=True, exist_ok=True)

    media_files: list[str] = []
    if audio_mode == "edge":
        items = collect_card_audio_items(cards)
        logger.info("Generating Edge TTS for %d unique clips (lemma + examples)", len(items))
        ensure_audio(items)
        paths = [path for _, path in items if is_valid_audio(path)]
        invalid = sum(1 for _, path in items if path.exists() and not is_valid_audio(path))
        if invalid:
            logger.warning("Skipping %d invalid/empty audio clips", invalid)
        ensure_anki_compatible(paths)
        media_files = sorted({str(path) for path in paths})
        logger.info("Bundling %d audio files into .apkg", len(media_files))
    elif audio_mode == "native":
        logger.info("Using device native TTS (%s) — no media bundled", ANKI_TTS_LANG)
    else:
        logger.info("Exporting text-only cards (no audio)")

    model = build_anki_model(audio_mode)
    deck = genanki.Deck(ANKI_DECK_ID, deck_name)

    field_fn = {
        "native": card_to_native_fields,
        "edge": card_to_edge_fields,
        "none": card_to_plain_fields,
    }[audio_mode]

    for card in cards:
        note = genanki.Note(model=model, fields=field_fn(card))
        deck.add_note(note)

    package = genanki.Package(deck)
    if media_files:
        package.media_files = media_files
    package.write_to_file(out)
    if audio_mode in ("native", "edge"):
        disable_apkg_autoplay(out)
        logger.info("Disabled automatic audio playback in deck options")
    logger.info("Exported %d cards to %s (%s audio)", len(cards), out, audio_mode)
    return out


def export_anki_csv(
    cards: list[dict] | None = None,
    output_path: str | None = None,
    excel_bom: bool = False,
) -> Path:
    """Write Anki-importable CSV."""
    if cards is None:
        cards = load_cards_jsonl()

    out = Path(output_path) if output_path else ANKI_CSV
    out.parent.mkdir(parents=True, exist_ok=True)

    encoding = "utf-8-sig" if excel_bom else "utf-8"
    with open(out, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for card in cards:
            writer.writerow(card_to_row(card))

    logger.info("Exported %d cards to %s", len(cards), out)
    return out


def run_export(
    input_path: str | None = None,
    output_path: str | None = None,
    excel_bom: bool = False,
    apkg: bool = False,
    apkg_path: str | None = None,
    audio_mode: str | None = None,
) -> Path:
    """Run export stage."""
    cards = load_cards_jsonl(input_path)
    export_anki_csv(cards, output_path, excel_bom=excel_bom)
    if apkg:
        mode = audio_mode or ANKI_AUDIO_MODE_DEFAULT
        return export_anki_apkg(cards, output_path=apkg_path, audio_mode=mode)
    return Path(output_path) if output_path else ANKI_CSV
