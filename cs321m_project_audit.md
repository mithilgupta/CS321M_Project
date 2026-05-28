# CS321M RAG Study — Project Audit & State of Play

**Document purpose:** Comprehensive audit of what the project is doing, what has been built, what the current state of the codebase is, and what remains. Written to bring a returning collaborator up to speed without assuming anything.

**Audit date:** May 2026  
**Auditor:** Claude (from direct inspection of terminal output, source files, config files, and project proposal document)

---

## 1. What This Project Is Actually About

### The Research Question

Enterprise RAG systems are typically evaluated against ad hoc question banks — collections of historical queries assembled without any principled measurement design. The problem is that these question banks were never designed as *measurement instruments*. As a result, evaluation scores may reflect artifacts of test composition rather than genuine system capability.

This project investigates whether you can transform an existing RAG question bank into a more valid internal benchmark, and whether you can fit calibrated difficulty and ability estimates under the sparse, low-compute conditions that enterprise settings impose.

This sits at the intersection of two fields:
- **RAG evaluation** — how do you measure whether a retrieval-augmented generation system is actually good?
- **Measurement science / psychometrics** — how do you design and calibrate a test instrument so that its scores mean something?

### The Two-Phase Plan

**Phase 1 — Audit the question bank**
- Score each query on a multi-dimensional technical difficulty taxonomy (via LLM-as-judge)
- Assign a manual business criticality label (informational → critical, 5 levels)
- Run embedding-based semantic coverage analysis over the document corpus
- Output: a structured gap analysis that makes the question bank diagnosable

**Phase 2 — Glicko-2/Elo Calibration**
- Run 100 annotated queries through 12 RAG system configurations
- Binarise RAGAS scores (pass/fail threshold)
- Fit a Glicko-2/Elo rating model to jointly estimate empirical difficulty per query and ability per system
- Output: per-query difficulty ratings and per-system ability ratings, each with uncertainty intervals

### Why Glicko-2/Elo Instead of IRT

Standard IRT (Rasch, 2PL) requires response matrices far larger than any enterprise setting can provide — typically hundreds of respondents and dozens of items. With 12 systems and 100 queries, joint IRT estimation is infeasible. The project surveyed five alternative approaches and selected Glicko-2/Elo as the primary method because:

- It assumes the least about the underlying data-generating process
- It requires no external calibration source
- It initialises item ratings from Phase 1 difficulty priors without treating them as fixed truth
- It is dimension-agnostic — the pass/fail criterion can be any external evaluation metric (faithfulness, factual correctness, hallucination rate, etc.) without modifying the calibration model

The secondary comparison baseline is fixed-item-parameter ability estimation (fix difficulty from Phase 1 heuristics, estimate only system ability via logistic likelihood).

---

## 2. The Dataset: OpenRAGBench

**Source:** Vectara's OpenRAGBench, downloaded from HuggingFace (`vectara/open_ragbench`)  
**Local path:** `data/raw/open_rag_bench/`

The benchmark contains:
- `corpus/` — scientific paper documents, each with title, abstract, sections, authors, categories
- `queries.json` — questions, each with `query`, `type` (extractive/abstractive), `source` (text/text-image/text-table/text-table-image)
- `qrels.json` — relevance judgments, each mapping a query ID to `doc_id` and `section_id`
- `answers.json` — reference answers keyed by query ID

**Scope decision made:** Text-only queries only (source == "text"). The benchmark is multimodal but full image/table support adds too much complexity for Phase 1. This is an explicit, documented simplification.

**Question subset:** 100 questions, 50 extractive + 50 abstractive, random seed 42, stored at:
`data/processed/subsets/openrag_text_only_100.json`

Each record in the subset has:
```
question_id, question, question_type, source,
gold_doc_id, gold_section_id, reference_answer
```

---

## 3. The Corpus Processing Pipeline

### What Was Built

**Section chunks** (`data/processed/corpus/section_chunks.jsonl`)
- Each document in `corpus/` is split into sections
- Tables and images are ignored (text only)
- Each chunk record contains:
  ```
  chunk_id (doc_id::sec::i), doc_id, section_id, title, abstract,
  authors, categories, published, text
  ```
- File size: ~112MB → large corpus

### Retrieval Indexes

**Dense index** (`data/indexes/dense/`)
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Each chunk is embedded as: `"Title: {title}\n\nAbstract: {abstract}\n\nSection: {text}"`
- Index: FAISS IndexFlatIP (inner product = cosine similarity on normalized embeddings)
- Files: `faiss.index` (28MB), `metadata.jsonl` (112MB), `embedder_name.txt`

**BM25 index** (`data/indexes/hybrid/`)
- Algorithm: BM25Okapi via `rank_bm25`
- Tokenization: simple lowercase whitespace split
- Files: `bm25.pkl` (95MB), `metadata.jsonl` (112MB)

---

## 4. The 12 RAG System Configurations

The experiment crosses three dimensions:

### Generators (3)
| Key | Provider | Model |
|-----|----------|-------|
| `strong_api` | OpenAI | gpt-4o |
| `mid_api` | OpenAI | gpt-4o-mini |
| `local_open` | HuggingFace local | Qwen/Qwen2.5-3B-Instruct |

### Retrievers (2)
| Key | Type | Description |
|-----|------|-------------|
| `dense` | Dense | FAISS inner product search |
| `hybrid` | Hybrid | 0.5 × normalized dense score + 0.5 × normalized BM25 score |

### Scaffolds (2)
| Key | top_k |
|-----|-------|
| `topk3` | 3 retrieved chunks |
| `topk8` | 8 retrieved chunks |

**Total combinations: 3 × 2 × 2 = 12 systems**

System IDs follow the pattern: `{generator}__{retriever}__{scaffold}`
Examples: `strong_api__dense__topk3`, `mid_api__hybrid__topk8`

---

## 5. The RAG Prompt

Every system uses the same prompt template (from `src/rag/generate.py`):

```
You are a question-answering assistant.

Use ONLY the retrieved context below to answer the question.
If the answer is not supported by the retrieved context, say:
"I do not have enough information from the retrieved context."

Retrieved context:
[Context 1]
Document Title: ...
Document ID: ...
Section ID: ...
Text: ...

[Context 2] ...

Question: {question}

Answer:
```

This is a standard closed-book RAG prompt — the model is instructed to use only the retrieved context.

---

## 6. The Output Format

Each system run produces a JSONL file at:
`outputs/runs/phase_1_1/{system_id}.jsonl`

Each line is one question-answer record with:
```json
{
  "question_id": "...",
  "question": "...",
  "question_type": "extractive|abstractive",
  "source": "text",
  "gold_doc_id": "...",
  "gold_section_id": 0,
  "reference_answer": "...",
  "system_id": "mid_api__dense__topk3",
  "generator_key": "mid_api",
  "generator_model": "gpt-4o-mini",
  "retriever_key": "dense",
  "retriever_type": "dense",
  "scaffold_key": "topk3",
  "top_k": 3,
  "retrieved_chunks": [...],
  "generated_answer": "...",
  "run_status": "success|error",
  "error": null
}
```

After all 12 runs, `src/runners/merge_runs.py` collapses everything into:
`outputs/merged/phase_1_1_all_runs.csv` (target: 1,200 rows = 100 questions × 12 systems)

---

## 7. Current State of the Project

### What Is Fully Done
- [x] OpenRAGBench downloaded
- [x] Section chunks built (112MB JSONL)
- [x] 100-question text-only subset created (50 extractive, 50 abstractive)
- [x] Dense FAISS index built
- [x] BM25 index built
- [x] All source code written and in place
- [x] All config files in place

### What Is Partially Done
- [~] One smoke-test run: `mid_api__dense__topk3` — **3 questions only, all errored**

### What Has Not Been Done
- [ ] The actual 12-system runs (all 100 questions each)
- [ ] Merged CSV
- [ ] RAGAS scoring (binarising answers into pass/fail)
- [ ] Phase 1 difficulty taxonomy scoring (LLM-as-judge)
- [ ] Business criticality labelling (manual)
- [ ] Semantic coverage analysis
- [ ] Glicko-2/Elo calibration
- [ ] Any analysis or reporting

### The Blocker

The OpenAI API key in `.env` has **exceeded its quota** (HTTP 429, `insufficient_quota`). This blocked all 3 smoke-test records.

**Note on the OpenAI API call:** `generate.py` uses `client.responses.create()` with `input=prompt` — this is the newer OpenAI Responses API, not the standard `chat.completions.create()`. This is intentional but worth knowing because the Responses API has slightly different behaviour and error handling than the Chat Completions API.

---

## 8. Key Design Decisions Worth Validating

Before proceeding, these are the decisions embedded in the current code that you should consciously agree with:

**1. Text-only subset**
Rationale: simplicity for Phase 1. The benchmark has multimodal queries but handling images/tables adds complexity. This limits generalisability to text-only RAG scenarios.

**2. Section-level chunking, no overlap**
Each corpus section is one chunk. No sliding window, no overlap. This means a question whose answer spans two sections will never be fully answered from a single retrieved chunk.

**3. all-MiniLM-L6-v2 as the dense embedder**
A small, fast model. Not state-of-the-art for retrieval (models like `bge-large` or `e5-mistral` would likely perform better). Chosen for speed and simplicity.

**4. Hybrid retrieval = simple linear interpolation**
0.5 × normalised dense + 0.5 × normalised BM25. No learned weights. The alpha values are hardcoded in `configs/retrievers.yaml`.

**5. Glicko-2/Elo over IRT**
Justified by the small response matrix (12 systems × 100 questions). This is a pragmatic choice, not a theoretically optimal one. Elo ratings are not unbiased IRT parameter estimates.

**6. RAGAS for scoring (not yet implemented)**
The plan is to binarise RAGAS scores at a threshold. RAGAS itself uses an LLM judge internally, which means scoring will cost additional API calls.

**7. The OpenAI Responses API**
`generate.py` uses `client.responses.create()` not `client.chat.completions.create()`. If switching to Anthropic, the generator class needs to be rewritten.

---

## 9. Recommended Next Steps

**Immediate (unblock the runs):**
1. Decide on API provider — OpenAI (add credits) or Anthropic (rewrite generator)
2. Run the full 12-system experiment (or a subset first to validate)
3. Merge all runs into the CSV

**Then Phase 1 analysis:**
4. Add RAGAS scoring to produce the binary response matrix
5. Run LLM-as-judge difficulty scoring on the 100 questions
6. Assign business criticality labels manually
7. Run semantic coverage analysis

**Then Phase 2:**
8. Fit Glicko-2/Elo model on the binary response matrix
9. Compare empirical difficulty estimates against Phase 1 priors
10. Report system ability rankings with uncertainty

---

## 10. File Map

```
cs321m-rag-study/
├── configs/
│   ├── experiment_phase_1_1.yaml   # ties together subset + generator/retriever/scaffold keys
│   ├── models.yaml                 # generator definitions (provider + model name)
│   ├── retrievers.yaml             # retriever types + alpha weights
│   └── scaffolds.yaml              # top_k values
├── data/
│   ├── raw/open_rag_bench/         # downloaded HuggingFace dataset
│   ├── processed/
│   │   ├── corpus/section_chunks.jsonl   # 112MB flat chunk file
│   │   └── subsets/openrag_text_only_100.json  # 100-question bank
│   └── indexes/
│       ├── dense/                  # FAISS index + metadata
│       └── hybrid/                 # BM25 pickle + metadata
├── outputs/
│   ├── runs/phase_1_1/             # one JSONL per system (only smoke test exists)
│   ├── merged/                     # empty — merge_runs.py not yet run
│   └── logs/                       # empty
└── src/
    ├── ingest/
    │   ├── download_openragbench.py
    │   ├── inspect_openragbench.py
    │   ├── build_section_chunks.py
    │   └── make_text_only_subset.py
    ├── rag/
    │   ├── build_dense_index.py
    │   ├── build_bm25_index.py
    │   ├── retrieve.py             # DenseRetriever + HybridRetriever classes
    │   └── generate.py             # OpenAIGenerator + HFLocalGenerator + build_prompt
    └── runners/
        ├── run_single_config.py    # main runner — takes --generator --retriever --scaffold
        └── merge_runs.py           # collapses all JSONL into one CSV
```

---

## 11. Open Questions for Validation

These are things that should be decided before proceeding, not assumed:

1. **API provider going forward** — OpenAI (add credits) or Anthropic? Or run local-only first?
2. **Is 100 questions still the right number?** The proposal says 100 but this could be adjusted.
3. **Is the 50/50 extractive/abstractive split intentional?** Or should the split reflect the natural distribution in the benchmark?
4. **RAGAS binarisation threshold** — the proposal says "determined from the observed score distribution." This needs a concrete decision rule before scoring.
5. **Should `strong_api` (gpt-4o) be replaced or supplemented with a Claude model?**
6. **Is the local model (Qwen 2.5 3B) appropriate?** It may be too small to produce meaningful answers, which would make the local configs uninteresting data points.
7. **What is the evaluation metric for RAGAS?** Faithfulness? Factual correctness? Answer relevancy? The Glicko model is dimension-agnostic but you need to pick one primary metric.

