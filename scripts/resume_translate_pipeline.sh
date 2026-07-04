#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
LOG="data/output/translate_pipeline.log"
mkdir -p data/output
{
  echo "=== $(date -Is) starting translate --top 10000 ==="
  .venv/bin/python -m src.main translate --top 10000
  echo "=== $(date -Is) starting filter ==="
  .venv/bin/python -m src.main filter
  echo "=== $(date -Is) starting rank --top 10000 ==="
  .venv/bin/python -m src.main rank --top 10000
  echo "=== $(date -Is) DONE ==="
  .venv/bin/python - <<'PY'
import json
import pandas as pd

d = len(json.load(open("data/reference/pt_es_dict.json")))
cleaned = pd.read_parquet("data/output/cleaned.parquet").head(10000)
scored = pd.read_parquet("data/output/scored.parquet")
ranked = pd.read_parquet("data/output/ranked.parquet")
scope = cleaned.merge(scored[["lemma", "penalty", "cognate_source"]], on="lemma")
cov = (scope["cognate_source"] == "translation").sum()
pen = (scope["penalty"] > 0).sum()
ranked[["freq_rank","lemma","pos","penalty","priority_score","es_translation","translation_similarity","cognate_source"]].to_csv("data/output/top10k_lemmas.csv", index=False)
print(f"dict={d} coverage={cov}/10000 penalized_in_top10k={pen}")
print("top10:", ranked[["lemma","penalty","es_translation"]].head(10).to_string(index=False))
PY
} >> "$LOG" 2>&1
