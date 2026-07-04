"""Step 4: LLM card generation via OpenRouter."""

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from src.cache import get_cached, set_cached
from src.config import (
    CARD_EXAMPLE_COUNT,
    CARDS_JSONL,
    MAX_CONCURRENT,
    MAX_RETRIES,
    CARD_GENERATION_MODEL,
    PROMPT_VERSION,
    RANKED_PARQUET,
    TEMPERATURE,
)
from src.models import VocabCard
from src.openrouter_client import get_client

load_dotenv()
logger = logging.getLogger(__name__)

POS_LABELS = {
    "nom": "noun",
    "ver": "verb",
    "adj": "adjective",
    "adv": "adverb",
}

EXAMPLE_SENTENCE_INSTRUCTIONS = """Generate exactly 5 Brazilian Portuguese example sentences.

The goal is not to demonstrate grammar. The goal is to help the learner naturally acquire the word through repeated exposure in realistic contexts.

Naturalness and word order:
- Prefer the most natural and common word order used by native Brazilians.
- Sound completely natural to native speakers; reflect modern everyday Brazilian Portuguese.
- Prioritize examples a Brazilian is genuinely likely to say or write over examples that merely illustrate a dictionary definition.
- Use mostly high-frequency vocabulary in the rest of the sentence.
- Be concise and natural. Most sentences should be approximately 5–15 words, but use whatever length sounds most natural.
- Be understandable without requiring previous context.
- Avoid obscure literary language, rare regional vocabulary, unnecessary slang, and proper names whenever possible.

Meaning coverage:
- When the lemma has multiple common meanings, cover them across the five examples — including different collocations, grammatical uses, or idiomatic expressions where relevant.
- When the lemma has essentially one core meaning, simply provide five natural, varied examples of everyday use; do not invent artificial sense distinctions.

Formatting:
- Contain exactly one occurrence of the target lemma (or one inflected form) per sentence.
- Wrap the target form with <b> and </b> tags.
- For each Portuguese sentence, provide one natural English translation.
- Do not provide Spanish sentence translations."""

SYSTEM_PROMPT = f"""You are a Brazilian Portuguese lexicographer creating vocabulary cards.

Audience: Spanish B2 speaker who already knows Portuguese grammar.
Output: valid JSON only, no markdown fences.

Card requirements:
- 1–4 concise English glosses (1–3 words each); list every common meaning you intend to illustrate
- 1–4 concise Spanish glosses (1–3 words each), Latin American Spanish
- Exactly {CARD_EXAMPLE_COUNT} example sentences following the rules below

{EXAMPLE_SENTENCE_INSTRUCTIONS}"""


def _card_cache_key(lemma: str, pos: str, model: str | None = None) -> tuple[str, ...]:
    return (lemma, pos, PROMPT_VERSION, model or CARD_GENERATION_MODEL)


def _build_user_prompt(lemma: str, pos: str, freq_rank: int) -> str:
    pos_label = POS_LABELS.get(pos, pos)
    return f"""Lemma: {lemma} ({pos_label}) — frequency rank {freq_rank}

Return JSON with this structure:
{{
  "lemma": "{lemma}",
  "part_of_speech": "{pos_label}",
  "english_translations": ["...", "..."],
  "spanish_translations": ["...", "..."],
  "examples": [
    {{"pt": "Eu <b>ontem</b> fui ao mercado.", "en": "I went to the market yesterday."}},
    ... exactly {CARD_EXAMPLE_COUNT} examples total
  ]
}}

If {lemma!r} has multiple common meanings, distribute examples across them. If it has one main meaning, write five natural everyday examples with varied contexts.

Each example has only "pt" and "en" fields. Each "pt" field must include exactly one <b>...</b> pair around the target word or an inflected form. Use natural Brazilian word order."""


def _parse_json_response(raw: str) -> object:
    """Parse LLM output, tolerating markdown fences and surrounding text."""
    text = raw.strip()
    if not text:
        raise ValueError("Empty LLM response")

    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _validate_card_content(card: VocabCard, lemma: str, pos: str) -> None:
    """Light post-validation — strict enough to catch obvious errors."""
    if card.lemma.lower() != lemma.lower():
        raise ValueError(f"Lemma mismatch: expected {lemma!r}, got {card.lemma!r}")

    if len(card.examples) != CARD_EXAMPLE_COUNT:
        raise ValueError(
            f"Expected {CARD_EXAMPLE_COUNT} examples, got {len(card.examples)}"
        )

    for i, ex in enumerate(card.examples):
        if ex.pt.count("<b>") != 1 or ex.pt.count("</b>") != 1:
            raise ValueError(
                f"Example {i + 1} must contain exactly one <b>...</b> pair: {ex.pt!r}"
            )
        if not re.search(r"<b>.+?</b>", ex.pt, re.IGNORECASE | re.DOTALL):
            raise ValueError(f"Example {i + 1} has empty or invalid <b> tags: {ex.pt!r}")
        if not ex.en.strip():
            raise ValueError(f"Example {i + 1} is missing an English translation")


def generate_card(
    client: OpenAI,
    lemma: str,
    pos: str,
    freq_rank: int,
    use_cache: bool = True,
    model: str | None = None,
) -> VocabCard:
    """Generate a single vocabulary card with retries."""
    model_name = model or CARD_GENERATION_MODEL
    cache_key = _card_cache_key(lemma, pos, model_name)
    if use_cache:
        cached = get_cached("card", *cache_key)
        if cached:
            card = VocabCard.model_validate(cached)
            _validate_card_content(card, lemma, pos)
            return card

    user_prompt = _build_user_prompt(lemma, pos, freq_rank)
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if last_error:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Previous attempt failed validation: {last_error}\n\n{user_prompt}",
                    }
                )
            else:
                messages.append({"role": "user", "content": user_prompt})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            data = _parse_json_response(raw)
            card = VocabCard.model_validate(data)
            _validate_card_content(card, lemma, pos)

            if use_cache:
                set_cached("card", card.model_dump(), *cache_key)
            return card

        except (json.JSONDecodeError, ValueError, Exception) as exc:
            last_error = str(exc)
            logger.warning(
                "Card generation attempt %d/%d failed for %s: %s",
                attempt,
                MAX_RETRIES,
                lemma,
                last_error,
            )

    raise RuntimeError(
        f"Failed to generate card for {lemma!r} after {MAX_RETRIES} attempts: {last_error}"
    )


def load_existing_cards(path: str | None = None) -> set[str]:
    """Return set of lemmas already in cards.jsonl."""
    jsonl_path = path or str(CARDS_JSONL)
    if not os.path.exists(jsonl_path):
        return set()
    done = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add(record.get("lemma", ""))
            except json.JSONDecodeError:
                continue
    return done


def append_card_record(
    card: VocabCard,
    metadata: dict,
    path: str | None = None,
) -> None:
    """Append a card record to jsonl."""
    jsonl_path = path or str(CARDS_JSONL)
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    record = {**card.model_dump(), **metadata}
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _generate_and_save_row(
    client: OpenAI,
    row: pd.Series,
    use_cache: bool,
    output_path: str | None = None,
    model: str | None = None,
) -> dict:
    """Generate one card and append immediately (partial progress on failure elsewhere)."""
    card = generate_card(
        client,
        row["lemma"],
        row["pos"],
        int(row["freq_rank"]),
        use_cache=use_cache,
        model=model,
    )
    metadata = {
        "freq_rank": int(row["freq_rank"]),
        "priority_score": float(row["priority_score"]),
        "pos_code": row["pos"],
        "model": model or CARD_GENERATION_MODEL,
    }
    append_card_record(card, metadata, path=output_path)
    return {**card.model_dump(), **metadata}


def generate_cards(
    df: pd.DataFrame,
    top_n: int | None = None,
    dry_run: bool = False,
    use_cache: bool = True,
    refresh_lemma: str | None = None,
    output_path: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> list[dict]:
    """Generate cards for ranked lemmas."""
    work = df.head(top_n) if top_n else df
    jsonl_path = output_path or str(CARDS_JSONL)
    existing = set() if force else load_existing_cards(jsonl_path)

    if refresh_lemma:
        existing.discard(refresh_lemma)

    tasks: list[pd.Series] = []
    for _, row in work.iterrows():
        lemma = row["lemma"]
        if lemma in existing and not refresh_lemma:
            continue
        if refresh_lemma and lemma != refresh_lemma:
            continue
        tasks.append(row)

    if dry_run:
        logger.info("Dry run: would generate %d cards", len(tasks))
        for row in tasks[:20]:
            logger.info("  - %s (%s) rank %s", row["lemma"], row["pos"], row["freq_rank"])
        if len(tasks) > 20:
            logger.info("  ... and %d more", len(tasks) - 20)
        return []

    if not tasks:
        logger.info("No new cards to generate")
        return []

    model_name = model or CARD_GENERATION_MODEL
    logger.info("Generating with model: %s → %s", model_name, jsonl_path)
    client = get_client()
    results: list[dict] = []
    failed: list[pd.Series] = []

    def _process(row: pd.Series) -> dict:
        return _generate_and_save_row(
            client, row, use_cache, output_path=jsonl_path, model=model
        )

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(_process, row): row for row in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating cards"):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error("Failed to generate card for %s: %s", row["lemma"], exc)
                failed.append(row)

    if failed:
        logger.info("Retrying %d failed cards sequentially", len(failed))
        for row in tqdm(failed, desc="Retrying failed"):
            try:
                results.append(
                    _generate_and_save_row(
                        client,
                        row,
                        use_cache=use_cache,
                        output_path=jsonl_path,
                        model=model,
                    )
                )
            except Exception as exc:
                logger.error("Retry failed for %s: %s", row["lemma"], exc)

    logger.info(
        "Generated %d/%d new cards (%d failed after retries)",
        len(results),
        len(tasks),
        len(tasks) - len(results),
    )
    return results


def run_generate(
    input_path: str | None = None,
    top_n: int | None = None,
    dry_run: bool = False,
    use_cache: bool = True,
    refresh_lemma: str | None = None,
    output_path: str | None = None,
    model: str | None = None,
    force: bool = False,
) -> list[dict]:
    """Run card generation stage."""
    path = input_path or str(RANKED_PARQUET)
    df = pd.read_parquet(path)
    return generate_cards(
        df,
        top_n=top_n,
        dry_run=dry_run,
        use_cache=use_cache,
        refresh_lemma=refresh_lemma,
        output_path=output_path,
        model=model,
        force=force,
    )
