# Project Architecture: Enterprise RAG Benchmark Validity
**Stanford CS321M — Phase 1 & Phase 2 Pipeline**

---

## Overview

This project evaluates the **validity of a RAG question bank** — not the RAG system itself.
It answers: *"Are these 100 questions a good measurement instrument?"*

The pipeline has two phases:

```
Phase 1: Question Bank Audit
    → Difficulty scoring (LLM-as-judge)
    → Semantic coverage (corpus coverage + query diversity)
    → Business criticality (LLM-as-judge)
    → RAG system runs + RAGAS scoring

Phase 2: Empirical Calibration
    → Glicko-2 ratings (question difficulty + system ability)
    → Validation: do Phase 1 scores predict Phase 2 outcomes?
```

---

## How a Single RAG Run Works

Take `mid_api__dense__topk3` as an example:

**Step 1 — Retrieve**
For each of the 100 questions, the Dense Retriever searches the FAISS
index and returns the top-k most semantically similar corpus chunks.

**Step 2 — Generate**
Those chunks get formatted into a prompt (via `build_prompt`) and sent
to the configured LLM. The model reads the context and generates an answer.

**Step 3 — Save**
The full record — question, retrieved chunks, generated answer, reference
answer — gets written as one JSON line to a `.jsonl` file.

---

## The 12 System Combinations

```
generator (3) × retriever (2) × scaffold (2) = 12 systems
```

| System ID | Generator | Model | Retriever | Top-K |
|---|---|---|---|---|
| strong_api__dense__topk3 | strong_api | claude-opus-4-5 | dense | 3 |
| strong_api__dense__topk8 | strong_api | claude-opus-4-5 | dense | 8 |
| strong_api__hybrid__topk3 | strong_api | claude-opus-4-5 | hybrid | 3 |
| strong_api__hybrid__topk8 | strong_api | claude-opus-4-5 | hybrid | 8 |
| mid_api__dense__topk3 | mid_api | claude-haiku-4-5 | dense | 3 |
| mid_api__dense__topk8 | mid_api | claude-haiku-4-5 | dense | 8 |
| mid_api__hybrid__topk3 | mid_api | claude-haiku-4-5 | hybrid | 3 |
| mid_api__hybrid__topk8 | mid_api | claude-haiku-4-5 | hybrid | 8 |
| local_open__dense__topk3 | local_open | Qwen2.5-3B | dense | 3 |
| local_open__dense__topk8 | local_open | Qwen2.5-3B | dense | 8 |
| local_open__hybrid__topk3 | local_open | Qwen2.5-3B | hybrid | 3 |
| local_open__hybrid__topk8 | local_open | Qwen2.5-3B | hybrid | 8 |

Each system produces one `.jsonl` file with 100 lines — one per question.

---

## How to Run a Single System

```bash
python -m src.runners.run_single_config \
  --generator mid_api \
  --retriever dense \
  --scaffold topk3
```

Output: `outputs/runs/phase_1_1/mid_api__dense__topk3.jsonl`

---

## What Each .jsonl Line Contains

```json
{
  "question_id"     : "36f030c0-...",
  "question"        : "Can spin-polarized STM...",
  "question_type"   : "extractive",
  "reference_answer": "Yes, spin-polarized STM...",
  "system_id"       : "mid_api__dense__topk3",
  "generator_model" : "claude-haiku-4-5",
  "retriever_type"  : "dense",
  "top_k"           : 3,
  "retrieved_chunks": [...],
  "generated_answer": "Based on the context...",
  "run_status"      : "success"
}
```

---

## After All 12 Runs — RAGAS Scoring

RAGAS reads each `.jsonl` record and computes two metrics:

- **Faithfulness** — is the generated answer grounded in the retrieved chunks?
- **Answer Relevance** — does the generated answer address the question?

Input: `question`, `generated_answer`, `retrieved_chunks`
Output: two scores per record, stored in `outputs/phase1/ragas/`

---

## After RAGAS — Glicko-2 Calibration

The 1,200 records (12 systems × 100 questions) get structured as a
competitive match matrix:

```
question_id | system_id                  | faithfulness | pass_fail
q_001       | strong_api__dense__topk3   |    0.87      |    1
q_001       | mid_api__dense__topk3      |    0.61      |    1
q_001       | local_open__dense__topk3   |    0.38      |    0
q_002       | strong_api__dense__topk3   |    0.71      |    1
...
```

Glicko-2 treats each (system, question) pair as a match:
- System wins if `pass_fail = 1` (RAGAS score ≥ threshold)
- Question wins if `pass_fail = 0`

After processing all 1,200 matches, Glicko-2 produces:
- **System ratings** — which configuration is empirically strongest?
- **Question difficulty ratings** — which questions defeat the most systems?

The key validation: do Glicko-2 question ratings correlate with
Phase 1 LLM-judged difficulty scores? If yes, the taxonomy is valid.

---

## Folder Structure

```
cs321m-rag-study/
│
├── configs/
│   ├── models.yaml              # generator definitions (Claude, Qwen)
│   ├── retrievers.yaml          # retriever definitions (dense, hybrid)
│   ├── scaffolds.yaml           # scaffold definitions (topk3, topk8)
│   └── experiment_phase_1_1.yaml# experiment config (which systems to run)
│
├── data/
│   ├── indexes/
│   │   ├── dense/               # FAISS index + metadata
│   │   └── hybrid/              # BM25 index
│   └── processed/
│       └── subsets/
│           └── openrag_text_only_100.json  # 100-question subset
│
├── src/
│   ├── rag/
│   │   ├── retrieve.py          # DenseRetriever, HybridRetriever
│   │   └── generate.py          # AnthropicGenerator, HFLocalGenerator
│   ├── runners/
│   │   └── run_single_config.py # runs one system config end-to-end
│   └── analysis/
│       ├── query_difficulty_diagnostic.py    # Phase 1: difficulty
│       ├── semantic_coverage_diagnostic.py   # Phase 1: coverage
│       ├── business_criticality_diagnostic.py# Phase 1: criticality
│       ├── ragas_eval.py                     # Phase 1: RAGAS scoring
│       └── prompts/
│           ├── query_formulation.txt
│           ├── reasoning_demand.txt
│           └── answer_form.txt
│
├── notebooks/
│   ├── phase1_semantic_coverage.ipynb  # coverage exploration
│   └── phase2_glicko.ipynb             # Glicko-2 calibration
│
└── outputs/
    ├── runs/
    │   └── phase_1_1/           # 12 .jsonl files (one per system)
    ├── phase1/
    │   ├── difficulty/          # LLM-judged difficulty scores
    │   ├── coverage/            # semantic coverage analysis
    │   ├── criticality/         # business criticality scores
    │   └── ragas/               # RAGAS scores
    └── phase2/
        └── glicko/              # Glicko-2 ratings + unified output
```

---

## Running the Full Pipeline

### Step 1: Run all 12 RAG systems
```bash
bash scripts/run_all_systems.sh
```

### Step 2: Score with RAGAS
```bash
python src/analysis/ragas_eval.py \
  --runs_dir outputs/runs/phase_1_1 \
  --output_dir outputs/phase1/ragas
```

### Step 3: Phase 1 diagnostics (can run in parallel)
```bash
# Terminal 1
python src/analysis/query_difficulty_diagnostic.py \
  --questions data/processed/subsets/openrag_text_only_100.json \
  --llm_provider anthropic \
  --output_dir outputs/phase1/difficulty

# Terminal 2
python src/analysis/business_criticality_diagnostic.py \
  --questions data/processed/subsets/openrag_text_only_100.json \
  --llm_provider anthropic \
  --output_dir outputs/phase1/criticality

# Terminal 3 (already done)
# semantic_coverage_diagnostic.py outputs already in outputs/phase1/coverage/
```

### Step 4: Phase 2 Glicko-2
```bash
jupyter lab notebooks/phase2_glicko.ipynb
```

---

## Key Design Decisions

**Why 12 systems?**
3 generators × 2 retrievers × 2 scaffolds gives enough variation to
produce meaningful Glicko-2 ratings. Stronger generators should rank
higher. Larger top-k should improve recall. Hybrid retrieval should
outperform dense-only on lexically direct questions.

**Why Glicko-2 over Elo?**
Glicko-2 adds Rating Deviation (RD) — an explicit uncertainty estimate
per question and per system. Questions with high RD are those where
system behavior was inconsistent (sometimes passing, sometimes failing).
These are the most discriminating questions for a valid benchmark.

**Why binarize RAGAS scores?**
Glicko-2 requires binary outcomes (win/loss). We binarize on a single
RAGAS metric (faithfulness by default) at a configurable threshold.
The threshold is a methodological choice — we report sensitivity to it.

**Why LLM-as-judge for Phase 1?**
Phase 1 diagnostic scores (difficulty, criticality) are computed from
query text alone — no corpus access required. This makes the framework
deployable as an automated pre-evaluation tool independent of any
specific RAG system.
