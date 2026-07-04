"""Step 5: Export Anki CSV and .apkg deck."""

import csv
import json
import logging
from pathlib import Path

import genanki

from src.config import ANKI_APKG, ANKI_CSV, CARD_EXAMPLE_COUNT, CARDS_JSONL
from src.tts import (
    card_audio_fields,
    collect_card_audio_items,
    ensure_audio,
)

logger = logging.getLogger(__name__)

# Stable IDs so re-exports update the same note type in Anki.
# Bumped when note fields/templates change (v3: front without example, random 2 on back).
ANKI_MODEL_ID = 1_985_092_385
ANKI_DECK_ID = 1_985_092_384
ANKI_DECK_NAME = "Brazilian Portuguese (ES speaker)"

ANKI_FIELD_NAMES = [
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
.sentence { line-height: 1.5; }
.gloss { margin-bottom: 0.8em; }
.gloss .es { color: #555; font-size: 0.95em; }
.examples { text-align: left; line-height: 1.45; }
.example { margin: 0.7em 0; padding-bottom: 0.5em; border-bottom: 1px solid #eee; }
.example:last-child { border-bottom: none; }
.en { color: #444; font-style: italic; }
"""

ANKI_FRONT = """
<div class="lemma">{{Lemma}} {{LemmaAudio}} <span class="pos">({{POS}})</span></div>
""".strip()

# JavaScript picks 2 distinct examples per review (reshuffles each time the card is shown).
ANKI_BACK = """
<div class="gloss">
<div><b>{{English}}</b></div>
<div class="es">{{Spanish}}</div>
</div>
<div id="pool" style="display:none">
<div class="ex">{{Ex1_PT}} {{Ex1_Audio}}<br><span class="en">{{Ex1_EN}}</span></div>
<div class="ex">{{Ex2_PT}} {{Ex2_Audio}}<br><span class="en">{{Ex2_EN}}</span></div>
<div class="ex">{{Ex3_PT}} {{Ex3_Audio}}<br><span class="en">{{Ex3_EN}}</span></div>
<div class="ex">{{Ex4_PT}} {{Ex4_Audio}}<br><span class="en">{{Ex4_EN}}</span></div>
<div class="ex">{{Ex5_PT}} {{Ex5_Audio}}<br><span class="en">{{Ex5_EN}}</span></div>
</div>
<div class="examples" id="show"></div>
<script>
(function () {
  var pool = Array.from(document.querySelectorAll("#pool .ex"));
  for (var i = pool.length - 1; i > 0; i--) {
    var j = Math.floor(Math.random() * (i + 1));
    var tmp = pool[i];
    pool[i] = pool[j];
    pool[j] = tmp;
  }
  var show = document.getElementById("show");
  pool.slice(0, 2).forEach(function (el) {
    var div = document.createElement("div");
    div.className = "example";
    div.innerHTML = el.innerHTML;
    show.appendChild(div);
  });
})();
</script>
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


def card_to_anki_fields(card: dict, audio: dict[str, str] | None = None) -> list[str]:
    """Convert a card record to genanki note fields."""
    row = card_to_row(card)
    audio = audio or {}
    return [
        row["Lemma"],
        row["PartOfSpeech"],
        row["EnglishTranslations"].replace(";", ", "),
        row["SpanishTranslations"].replace(";", ", "),
        *[row[f"Sentence{i}_PT"] for i in range(1, CARD_EXAMPLE_COUNT + 1)],
        *[row[f"Sentence{i}_EN"] for i in range(1, CARD_EXAMPLE_COUNT + 1)],
        audio.get("LemmaAudio", ""),
        *[audio.get(f"Ex{i}_Audio", "") for i in range(1, CARD_EXAMPLE_COUNT + 1)],
    ]


def build_anki_model() -> genanki.Model:
    """Return the shared PT-BR vocabulary note type."""
    return genanki.Model(
        ANKI_MODEL_ID,
        "PT-BR Vocab (ES speaker)",
        fields=[{"name": name} for name in ANKI_FIELD_NAMES],
        templates=[
            {
                "name": "PT → EN",
                "qfmt": ANKI_FRONT,
                "afmt": ANKI_BACK,
            }
        ],
        css=ANKI_CSS,
    )


def export_anki_apkg(
    cards: list[dict] | None = None,
    output_path: str | None = None,
    deck_name: str = ANKI_DECK_NAME,
    with_audio: bool = True,
) -> Path:
    """Write a ready-to-import .apkg file with note type and deck."""
    if cards is None:
        cards = load_cards_jsonl()

    out = Path(output_path) if output_path else ANKI_APKG
    out.parent.mkdir(parents=True, exist_ok=True)

    media_files: list[str] = []
    if with_audio:
        items = collect_card_audio_items(cards)
        logger.info("Generating TTS for %d unique clips (lemma + examples)", len(items))
        ensure_audio(items)
        media_files = sorted({str(path) for _, path in items if path.exists()})
        logger.info("Bundling %d audio files into .apkg", len(media_files))

    model = build_anki_model()
    deck = genanki.Deck(ANKI_DECK_ID, deck_name)

    for card in cards:
        audio = card_audio_fields(card) if with_audio else {}
        note = genanki.Note(model=model, fields=card_to_anki_fields(card, audio))
        deck.add_note(note)

    package = genanki.Package(deck)
    if media_files:
        package.media_files = media_files
    package.write_to_file(out)
    logger.info("Exported %d cards to %s", len(cards), out)
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
    with_audio: bool = True,
) -> Path:
    """Run export stage."""
    cards = load_cards_jsonl(input_path)
    export_anki_csv(cards, output_path, excel_bom=excel_bom)
    if apkg:
        return export_anki_apkg(cards, output_path=apkg_path, with_audio=with_audio)
    return Path(output_path) if output_path else ANKI_CSV
