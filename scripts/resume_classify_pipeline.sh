#!/usr/bin/env bash
# Resume or finish classify → filter → rank pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."

.venv/bin/python -m src.main classify --top 10000
.venv/bin/python -m src.main filter
.venv/bin/python -m src.main rank --top 10000

.venv/bin/python << 'PY'
import pandas as pd
ranked = pd.read_parquet("data/output/ranked.parquet")
out = ranked.copy()
out.insert(0, "deck_rank", range(1, len(out) + 1))
cols = [
    "deck_rank", "freq_rank", "lemma", "pos", "es_translation",
    "translation_similarity", "dissimilarity_factor", "study_weight",
    "priority_score", "studyable", "exclude_reason", "cognate_source",
]
study = out[[c for c in cols if c in out.columns]]
study.to_csv("data/output/top10k_lemmas.csv", index=False)
study.to_excel("data/output/ranked.xlsx", index=False, sheet_name="ranked")
print(f"Deck: {len(study)} lemmas")
for w in ["shopping", "ontem", "dizer", "black", "boom"]:
    print(f"  {w}: {'in deck' if w in set(ranked.lemma) else 'excluded'}")
PY
