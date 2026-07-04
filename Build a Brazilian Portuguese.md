# Build a Brazilian Portuguese Anki Deck Generator (High-Quality Pipeline)

You are an expert Python engineer and computational linguist.

Build a **production-quality Python project** that generates a high-quality Anki deck for learning Brazilian Portuguese vocabulary.

The system is designed for:

* A Spanish speaker (B2 level)
* Already knows Portuguese grammar
* Wants maximum vocabulary efficiency
* Wants to avoid wasting time on obvious Portuguese–Spanish cognates

---

# Core Principle

We are NOT building a dictionary.

We are building a **learning-optimized vocabulary sequence**.

Each lemma should represent a **high-value learning unit**.

---

# Input Data

You will be given a file:

### Brazilian Portuguese Lemma Frequency List

It contains:

* lemma (dictionary form)
* frequency rank
* frequency count
* noise entries (numbers, symbols, broken tokens)

Example cleanup needed:

* remove numbers
* remove punctuation-only entries
* remove corrupted tokens
* normalize spacing and encoding

---

# Step 1: Data Cleaning

Clean the frequency list:

Remove:

* numbers
* symbols
* punctuation-only tokens
* empty rows
* malformed Unicode
* non-lexical entries

Keep only valid Portuguese lemmas.

---

# Step 2: Cognate Filtering (IMPORTANT)

We do NOT want false friend detection.

We only want to REMOVE obvious Spanish cognates that are already effectively known.

Goal:

> Avoid wasting study time on words that are nearly identical to Spanish.

Examples to de-prioritize or remove:

* hospital
* problema
* importante
* diferente
* animal

Examples to keep:

* achar
* ficar
* puxar
* pegar
* jeito
* saudade

Method:

You must design a practical heuristic using:

* normalized string similarity (RapidFuzz or Levenshtein)
* accent stripping
* optional comparison against a Spanish lemma list (if useful)

BUT:

* Do NOT over-filter
* Do NOT remove useful but similar words blindly

You must propose and justify the best approach before implementing it.

---

# Step 3: Final Ranked List

Produce a ranked list of lemmas ordered by:

PriorityScore = frequency adjusted by cognate penalty

---

# Step 4: Card Generation (LLM-based, critical step)

For each lemma, call an LLM to generate structured output.

You MUST generate ALL of the following in a single structured response per lemma:

## Output schema (JSON)

{
"lemma": "",
"part_of_speech": "",

"english_translations": [
"1–4 concise meanings (1–3 words each)"
],

"spanish_translations": [
"1–4 concise meanings (1–3 words each)"
],

"examples": [
{
"pt": "",
"en": "",
"es": ""
},
...
up to 4 examples
]
}

---

# Example Sentence Requirements

For each lemma:

* 3–4 sentences
* natural Brazilian Portuguese
* modern usage
* simple grammar (A2–B1 level)
* short (ideally < 12–15 words)
* each sentence should illustrate a common usage pattern
* avoid names unless necessary
* avoid rare slang unless the word itself is slang
* avoid overly literary or academic style

Sentences must be consistent with the chosen meanings.

---

# Step 5: Output Format

Generate a CSV suitable for Anki import.

One row per lemma.

Columns:

* Lemma
* PartOfSpeech
* FrequencyRank
* EnglishTranslations (semicolon-separated)
* SpanishTranslations (semicolon-separated)
* Sentence1_PT
* Sentence1_EN
* Sentence1_ES
* Sentence2_PT
* Sentence2_EN
* Sentence2_ES
* Sentence3_PT
* Sentence3_EN
* Sentence3_ES
* Sentence4_PT
* Sentence4_EN
* Sentence4_ES
* PriorityScore

---

# Step 6: Audio

Do NOT generate audio.

Audio will be handled later inside Anki using HyperTTS or AwesomeTTS.

---

# Step 7: Engineering Requirements

* Use Python 3.11+
* Modular architecture
* Clean separation of concerns

Suggested structure:

src/
main.py
clean.py
filter.py
rank.py
llm.py
export.py
config.py

data/
input/
output/

* Use pandas where appropriate
* Use RapidFuzz for similarity
* Use tqdm for progress
* Use pathlib
* Add logging

---

# Step 8: Extensibility

Design system so future upgrades are easy:

* adding audio fields
* adding CEFR level tagging
* adding synonym expansion
* adding image support
* swapping LLM provider
* caching LLM responses per lemma

---

# Step 9: Critical Design Rule

Before writing code:

1. Propose architecture
2. Explain cognate filtering strategy
3. Explain how LLM calls will be structured
4. Explain how caching will work
5. Explain risks and failure modes

ONLY after approval:

Start coding one file at a time.

Wait for approval between files.

---

# Success Criteria

The final output should:

* produce a high-quality Anki deck
* minimize wasted reviews
* avoid obvious Spanish redundancy
* use natural Brazilian Portuguese examples
* be robust and reproducible

