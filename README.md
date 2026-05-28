# Toward Valid Internal Benchmarks for Enterprise RAG Evaluation
**Stanford CS321M — AI Measurement Science**
**Author:** Mithil Gupta (mithilg@stanford.edu)

---

## Overview

This repository implements a two-phase measurement-science framework for
auditing enterprise RAG question banks as measurement instruments.

**Phase 1 — Question Bank Audit:** Scores each query on three validity
dimensions simultaneously:
- Query difficulty (LLM-as-judge, literature-grounded taxonomy)
- Business criticality (LLM-as-judge, domain-criticality map)
- Semantic corpus coverage (UMAP + HDBSCAN clustering)

**Phase 2 — Empirical Calibration:** Runs 100 queries through 12 RAG
system configurations, binarizes evaluation scores, and fits a Glicko-2/Elo
rating model to jointly estimate empirical question difficulty and system ability.

---

## Repository Structure

```
cs321m-rag-study/
├── configs/
│   ├── models.yaml                  # Generator definitions (Claude, Qwen)
│   ├── retrievers.yaml              # Retriever definitions (dense, hybrid)
│   ├── scaffolds.yaml               # Scaffold definitions (topk3, topk8)
│   └── experiment_phase_1_1.yaml   # Experiment config
│
├── data/
│   └── processed/
│       └── subsets/
│           └── openrag_text_only_100.json  # 100-question subset
│
├── src/
│   ├── rag/
│   │   ├── retrieve.py              # DenseRetriever, HybridRetriever
│   │   └── generate.py              # AnthropicGenerator, HFLocalGenerator
│   ├── runners/
│   │   └── run_single_config.py     # Runs one RAG system config end-to-end
│   └── analysis/
│       ├── query_difficulty_diagnostic.py     # Phase 1: difficulty scoring
│       ├── business_criticality_diagnostic.py # Phase 1: criticality scoring
│       ├── semantic_coverage_diagnostic.py    # Phase 1: corpus coverage
│       ├── ragas_score.py                     # Phase 2: real RAGAS scoring
│       ├── ragas_score_mock.py                # Phase 2: synthetic RAGAS scores
│       └── prompts/
│           ├── query_formulation.txt          # LLM prompt: formulation difficulty
│           ├── reasoning_demand.txt           # LLM prompt: reasoning demand
│           └── answer_form.txt                # LLM prompt: answer form complexity
│
├── notebooks/
│   ├── phase1_semantic_coverage.ipynb   # Coverage analysis exploration
│   └── phase2_glicko.ipynb              # Glicko-2 calibration
│
├── scripts/
│   ├── run_all_systems.sh               # Batch runner for 12 RAG systems
│   └── download_modal_results.sh        # Downloads results from Modal volume
│
├── outputs/
│   ├── phase1/
│   │   ├── difficulty/      # LLM-judged difficulty scores + report
│   │   ├── coverage/        # Semantic coverage analysis + visualizations
│   │   ├── criticality/     # Business criticality scores + report
│   │   └── ragas/           # RAGAS scores + report
│   └── phase2/              # Glicko-2 ratings + unified output
│
├── run_modal.py             # Modal cloud runner for 12 RAG systems
├── ARCHITECTURE.md          # Full pipeline architecture documentation
├── requirements.txt         # Python dependencies (Phase 1 + RAG pipeline)
├── requirements_ragas.txt   # Python dependencies (RAGAS scoring, Python 3.11)
└── .env.example             # Environment variable template
```

---

## Environment Setup

### Prerequisites
- Python 3.13 (main environment)
- Python 3.11 (RAGAS scoring only — see note below)
- Anthropic API key
- Conda (recommended)

### Step 1 — Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/cs321m-rag-study.git
cd cs321m-rag-study

# Main environment (Python 3.13)
conda create -n cs321m python=3.13 -y
conda activate cs321m
pip install -r requirements.txt
```

### Step 2 — Set up API keys

```bash
cp .env.example .env
# Edit .env and add your Anthropic API key
```

### Step 3 — RAGAS environment (Python 3.11 required)

RAGAS 0.1.21 is incompatible with Python 3.13 due to a langchain-community
dependency issue. A separate environment is required for real RAGAS scoring.

```bash
conda create -n cs321m_ragas python=3.11 -y
conda activate cs321m_ragas
pip install -r requirements_ragas.txt
```

---

## Reproducing Results

### Phase 1: Question Bank Audit

Run all three Phase 1 diagnostics. They are independent and can run in parallel
in separate terminals.

```bash
conda activate cs321m
export ANTHROPIC_API_KEY="your-key-here"

# Terminal 1: Query difficulty (300 API calls, ~15 min, ~$1.40)
python src/analysis/query_difficulty_diagnostic.py \
  --questions data/processed/subsets/openrag_text_only_100.json \
  --prompts_dir src/analysis/prompts \
  --llm_provider anthropic \
  --output_dir outputs/phase1/difficulty

# Terminal 2: Business criticality (100 API calls, ~5 min, ~$0.50)
python src/analysis/business_criticality_diagnostic.py \
  --questions data/processed/subsets/openrag_text_only_100.json \
  --llm_provider anthropic \
  --output_dir outputs/phase1/criticality

# Terminal 3: Semantic coverage (no API calls, ~10 min compute)
python src/analysis/semantic_coverage_diagnostic.py \
  --corpus_embeddings outputs/phase1/coverage/corpus_embeddings.npy \
  --questions data/processed/subsets/openrag_text_only_100.json \
  --metadata data/indexes/dense/metadata.jsonl \
  --embedder_name data/indexes/dense/embedder_name.txt \
  --output_dir outputs/phase1/coverage
```

### Phase 2: RAG System Runs

#### Option A — Run locally (Claude systems only, Qwen may OOM)

```bash
conda activate cs321m
export ANTHROPIC_API_KEY="your-key-here"
bash scripts/run_all_systems.sh
```

#### Option B — Run on Modal (recommended, all 12 systems in parallel)

```bash
# One-time setup
pip install modal
modal setup
modal secret create anthropic-secret ANTHROPIC_API_KEY=your-key-here

# Run all 12 systems in parallel
modal run run_modal.py

# Download results
bash scripts/download_modal_results.sh
```

### Phase 2: RAGAS Scoring

#### Real RAGAS scores (requires cs321m_ragas environment)

```bash
conda activate cs321m_ragas
export ANTHROPIC_API_KEY="your-key-here"
python src/analysis/ragas_score.py \
  --runs_dir outputs/runs/phase_1_1 \
  --output_dir outputs/phase1/ragas
```

#### Synthetic RAGAS scores (no API needed, for pipeline demonstration)

```bash
conda activate cs321m
python src/analysis/ragas_score_mock.py \
  --runs_dir outputs/runs/phase_1_1 \
  --difficulty outputs/phase1/difficulty/per_query_scores.json \
  --output_dir outputs/phase1/ragas
```

### Phase 2: Glicko-2 Calibration

```bash
conda activate cs321m
jupyter lab notebooks/phase2_glicko.ipynb
```

Run all cells in order. The notebook reads from `outputs/phase1/ragas/per_query_scores.json`
and writes to `outputs/phase2/`.

---

## Paper Figures

| Figure | Script | Output path |
|---|---|---|
| Fig 1: Difficulty distribution | `query_difficulty_diagnostic.py` | `outputs/phase1/difficulty/distribution_overall.png` |
| Fig 2: Per-dimension scores | `query_difficulty_diagnostic.py` | `outputs/phase1/difficulty/distribution_by_dimension.png` |
| Fig 3: UMAP coverage map | `semantic_coverage_diagnostic.py` | `outputs/phase1/coverage/umap_2d_coverage.png` |
| Fig 4: RAGAS by system | `ragas_score_mock.py` | `outputs/phase1/ragas/ragas_by_system.png` |

---

## Expected Runtime and Compute

| Component | Runtime | API Cost | Environment |
|---|---|---|---|
| Query difficulty scoring | ~15 min | ~$1.40 | cs321m |
| Business criticality scoring | ~5 min | ~$0.50 | cs321m |
| Semantic coverage analysis | ~10 min | $0 | cs321m |
| RAG system runs (Modal) | ~45 min | ~$4.00 | Modal |
| RAGAS scoring (real) | ~60 min | ~$0.60 | cs321m_ragas |
| RAGAS scoring (mock) | <1 min | $0 | cs321m |
| Glicko-2 calibration | <5 min | $0 | cs321m |

---

## Dataset

This project uses **OpenRAGBench** by Vectara:
- HuggingFace: https://huggingface.co/datasets/vectara/open_ragbench
- We use a 100-question text-only subset (50 extractive, 50 abstractive, seed=42)
- The subset is included at `data/processed/subsets/openrag_text_only_100.json`

The FAISS dense index (18,840 chunks) is not included in the repository due
to size (>1GB). To rebuild it, follow the ingestion instructions in
`src/ingest/`.

---

## Key Design Decisions

- **UMAP + HDBSCAN over k-means:** Eliminates manual cluster-count selection;
  handles non-spherical cluster structure; explicit noise modeling
- **Separate prompts per difficulty dimension:** Prevents dimension conflation
- **Temperature=0 for all LLM judges:** Maximizes intra-judge consistency
- **Fixed binarization threshold (0.75):** Avoids circular reasoning from
  data-adaptive thresholds
- **Glicko-2 initialized from Phase 1 scores:** Phase 1 priors are testable
  against empirical outcomes

---

## Notes on Reproducibility

- All random seeds are set to 42
- RAGAS scores in the paper are synthetic (see `ragas_score_mock.py`)
- Real RAGAS scoring requires API access and the `cs321m_ragas` environment
- The `.env` file is not included — copy `.env.example` and add your keys
