# Brazilian Portuguese Anki Deck Generator

Production pipeline that builds a learning-optimized Brazilian Portuguese vocabulary deck for **Spanish B2 speakers** who already know Portuguese grammar.

## Features

- Cleans LexPorBR frequency data (content words + heuristic noise filter)
- **LLM studyability classifier** — excludes foreign loans, acronyms, corpus junk
- **Translation-first cognate filtering** — PT lemma vs Spanish gloss similarity
- Ranks lemmas by `PriorityScore = log10(freq) × (1 - cognate_penalty)`
- Generates Anki cards via **Claude Sonnet 4.6** on **OpenRouter** (translate/classify use DeepSeek)
- Disk caching and resumable output

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env
# Add OPENROUTER_API_KEY for translate + generate steps
```

## Quick start

```bash
# Full pipeline (test with 100 lemmas, no API for cards)
python -m src.main run-all --top 100 --dry-run

# With translation + card generation (requires API key)
python -m src.main run-all --top 100

# Production deck (10,000 lemmas; translates top 25k by default)
python -m src.main run-all --top 10000
```

## CLI commands

| Command | Description |
|---------|-------------|
| `prepare-data` | Write seed PT→ES dictionary (~170 high-frequency entries) |
| `clean` | Clean LexPorBR input |
| `classify` | LLM studyability labels → `clean_labels.json` |
| `translate` | LLM batch PT→ES translation → `pt_es_dict.json` |
| `filter` | Cognate scoring via translation similarity |
| `rank` | Priority ranking |
| `generate` | LLM card generation |
| `export` | Anki CSV export |
| `run-all` | Full pipeline |

### Useful flags

- `--top N` — limit deck to top N ranked lemmas
- `--skip-classify` — reuse existing `clean_labels.json`
- `--skip-translate` — reuse existing `pt_es_dict.json`
- `--dry-run` — preview without API calls
- `--no-cache` — bypass LLM cache
- `--refresh-lemma X` — retranslate or regenerate one lemma

## Cognate filtering (translation-first)

```text
1. Translate PT lemma → Spanish gloss(es)  [translate.py + LLM]
2. CognateScore = max `cognate_similarity(PT_lemma, ES_gloss)` — PT→ES orthographic normalization, Jaro-Winkler for vowel-shift cognates, suffix-overlap guard
3. penalty = 1.0 when CognateScore ≥ 75 (excluded from deck)
4. PriorityScore = log10(freq) × (1 - penalty)
```

Only words whose **Spanish translation looks like the Portuguese form** are deprioritized (e.g. `hospital`/`hospital`). Words with different translations stay (`ficar`/`quedar`, `achar`/`encontrar`).

## Pipeline

```
LexPorBR → clean → classify → translate → filter → rank → generate → export
```

## Project structure

```
src/
  classify.py       LLM studyability classifier + cache
  translate.py      PT→ES LLM translation + cache
  filter.py         translation-first cognate scoring
  seed_dict.py      bootstrap translations for smoke tests
  seed_labels.py    bootstrap studyability labels for smoke tests
  clean.py, rank.py, llm.py, export.py, main.py
data/
  reference/pt_es_dict.json, clean_labels.json
  output/cleaned.parquet, scored.parquet, ranked.parquet, anki_deck.csv
```

## Testing workflow

```bash
python -m src.main prepare-data
python -m src.main clean
python -m src.main classify --top 100    # or --dry-run first
python -m src.main translate --top 500    # or --dry-run first
python -m src.main filter
python -m src.main rank --top 100
python -m src.main generate --top 10    # smoke test
python -m src.main run-all --top 10000  # production
```
