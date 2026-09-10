# Hiver Support Agent

AI customer-support agent prototype built on the Customer Support on Twitter
dataset: intent classification, grounded (RAG) reply generation, and a
deterministic + model-informed human-escalation decision.

**Status: Phase 1 (Data) complete.** Intent classification, retrieval,
agent, and evaluation phases are being built next — this README will grow
with each phase.

## 1. Project overview

The system takes an inbound customer message and returns:
- a classified **intent** (from a taxonomy derived from the selected
  brand's real historical data — not a canned list),
- a **draft reply** grounded in how that brand actually handled similar
  issues historically, and
- an **AUTO_HANDLE / ESCALATE** decision with an explicit reason.

## 2. Architecture (Phase 1 slice)

```
raw tweets (CSV)
   ↓ src/data/loader.py       — schema-validated load
   ↓ src/data/cleaner.py      — dedupe, drop empty/spam/malformed rows
   ↓ src/data/thread_builder.py — reconstruct conversations from reply chains
   ↓ src/data/brand_selector.py — rank brands, recommend one with justification
```

The rest of the pipeline (intent taxonomy, retrieval/RAG, agent, evaluation)
follows the architecture described in the assignment spec and will be added
phase by phase.

## 3. Prerequisites

- Python 3.11+
- (For later phases) an OpenAI API key for LLM + embeddings

## 4. Installation

```bash
git clone <repo>
cd hiver-support-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Dataset setup

This repo ships a small `data/raw/sample.csv` (~100 rows) with the real
column schema of the Kaggle `thoughtvector/customer-support-on-twitter`
dataset, so the pipeline runs out of the box without downloading anything.

To use the full dataset:
1. Download `twcs.csv` from Kaggle (`thoughtvector/customer-support-on-twitter`).
2. Place it at `data/raw/twcs.csv`.
3. Run the pipeline with `--raw-path data/raw/twcs.csv --max-rows 50000`
   (or your preferred cap — see `dataset.sample_size` in
   `configs/config.yaml`). **You do not need to process the full ~3M rows**
   to reproduce headline results.

## 6. Environment variables

Copy `.env.example` to `.env` and fill in your API key (needed starting in
the intent/retrieval/agent phases, not for Phase 1):

```bash
cp .env.example .env
```

Never commit `.env`.

## 7. Data preparation (Phase 1)

```bash
python scripts/prepare_data.py --raw-path data/raw/sample.csv
```

This loads the raw CSV (schema-validated against the columns documented in
`src/data/loader.py`), cleans it, and reconstructs conversation threads by
walking the tweet reply chain. Outputs land in `data/processed/`:

- `cleaned_tweets.csv`
- `conversations.json` — normalized `{conversation_id, brand, messages[]}` records
- `cleaning_report.json`, `thread_build_report.json` — counts of what was removed/dropped and why

On the sample dataset:
```
Loaded 93 rows
Cleaned: 0 removed (duplicates/empty/malformed/mention-only spam)
Threads: 25 root customer tweets → 24 valid conversations
(1 root tweet dropped: no agent reply found)
```

## 8. Brand selection

```bash
python scripts/select_brand.py
```

Analyzes `data/processed/conversations.json` and ranks brands by a
composite score (conversation volume, unique threads, average thread
depth, and a resolution-rate heuristic — see `src/data/brand_selector.py`
docstring and `DECISION_LOG.md` for the exact weighting and rationale).
Writes `data/processed/brand_ranking.csv`.

**On the current sample data, `AppleSupport` is the top-ranked brand**
(11 conversations, composite score 0.70) — but this sample is only ~100
rows, so this ranking is illustrative, not a real brand decision. This
will be re-run against a larger sample before the brand choice is
finalized in the report (flagged in `DECISION_LOG.md` item 6).

## 9. Running tests (Phase 1)

```bash
pytest tests/test_cleaner.py tests/test_thread_builder.py
```

(This dev sandbox has no `pytest` installed — tests were verified by
running each test function directly: `python tests/test_cleaner.py` and
`python tests/test_thread_builder.py`. Both currently pass, 10/10.)

## 10. Project structure

```
hiver-support-agent/
├── README.md
├── DECISION_LOG.md
├── requirements.txt
├── .env.example
├── configs/config.yaml
├── data/{raw,processed,golden,sample}/
├── src/
│   ├── config.py
│   └── data/{loader,cleaner,thread_builder,brand_selector}.py
├── scripts/{prepare_data,select_brand}.py
└── tests/{test_cleaner,test_thread_builder}.py
```

(`src/intents/`, `src/retrieval/`, `src/agent/`, `src/llm/`,
`src/evaluation/` exist as empty package stubs, to be filled in the
next phases.)

## 11. Index creation, running the API, evaluation, results, failure
analysis, limitations

Not yet implemented — coming in later phases.
