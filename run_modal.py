"""
run_modal.py
=============
Modal launcher for RAG batch evaluation.
Runs all 12 system configurations in parallel on Modal cloud.

Claude systems (strong_api, mid_api) run on CPU — no GPU needed.
Qwen system (local_open) runs on T4 GPU.

USAGE:
    # Run all 12 systems (full 100 questions each)
    modal run run_modal.py

    # Test with 3 questions per system
    modal run run_modal.py --limit 3

    # Run a single system
    modal run run_modal.py --system strong_api__dense__topk3

RESULTS:
    Results are saved to Modal Volume: rag-eval-outputs
    Download with:
        modal volume get rag-eval-outputs <system_id>.jsonl outputs/runs/phase_1_1/<system_id>.jsonl

    Or download all at once:
        bash scripts/download_modal_results.sh
"""

import modal
import json
from pathlib import Path

# ── MODAL APP ─────────────────────────────────────────────────────────────────
app = modal.App("cs321m-rag-eval")

# Volume to persist results across runs
volume = modal.Volume.from_name(
    "rag-eval-outputs",
    create_if_missing=True,
)

# ── BASE IMAGE (CPU — for Claude API systems) ─────────────────────────────────
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "anthropic",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "rank-bm25",
        "pyyaml",
        "tqdm",
        "python-dotenv",
    ])
    # Mount all source code
    .add_local_dir("src",     remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    # Mount FAISS index and metadata
    .add_local_dir("data/indexes",   remote_path="/root/data/indexes")
    .add_local_dir("data/processed", remote_path="/root/data/processed")
)

# ── GPU IMAGE (for Qwen local_open systems) ───────────────────────────────────
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "transformers",
        "accelerate",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "rank-bm25",
        "pyyaml",
        "tqdm",
        "python-dotenv",
    ])
    .add_local_dir("src",     remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_dir("data/indexes",   remote_path="/root/data/indexes")
    .add_local_dir("data/processed", remote_path="/root/data/processed")
)

# ── SYSTEM DEFINITIONS ────────────────────────────────────────────────────────
ALL_SYSTEMS = [
    {"generator": "strong_api", "retriever": "dense",   "scaffold": "topk3"},
    {"generator": "strong_api", "retriever": "dense",   "scaffold": "topk8"},
    {"generator": "strong_api", "retriever": "hybrid",  "scaffold": "topk3"},
    {"generator": "strong_api", "retriever": "hybrid",  "scaffold": "topk8"},
    {"generator": "mid_api",    "retriever": "dense",   "scaffold": "topk3"},
    {"generator": "mid_api",    "retriever": "dense",   "scaffold": "topk8"},
    {"generator": "mid_api",    "retriever": "hybrid",  "scaffold": "topk3"},
    {"generator": "mid_api",    "retriever": "hybrid",  "scaffold": "topk8"},
    {"generator": "local_open", "retriever": "dense",   "scaffold": "topk3"},
    {"generator": "local_open", "retriever": "dense",   "scaffold": "topk8"},
    {"generator": "local_open", "retriever": "hybrid",  "scaffold": "topk3"},
    {"generator": "local_open", "retriever": "hybrid",  "scaffold": "topk8"},
]

def system_id(cfg):
    return f"{cfg['generator']}__{cfg['retriever']}__{cfg['scaffold']}"


# ── CORE RUN FUNCTION (CPU — Claude systems) ──────────────────────────────────
@app.function(
    image   = cpu_image,
    timeout = 3600,
    secrets = [modal.Secret.from_name("anthropic-secret")],
    volumes = {"/root/outputs": volume},
)
def run_claude_system(generator: str, retriever: str, scaffold: str, limit: int = None) -> dict:
    import sys
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import os
    sys.path.insert(0, "/root")

    from src.rag.retrieve import DenseRetriever, HybridRetriever
    from src.rag.generate import AnthropicGenerator, build_prompt
    import yaml
    import json
    from tqdm import tqdm
    from pathlib import Path

    sid = f"{generator}__{retriever}__{scaffold}"
    print(f"[modal] Starting: {sid}")

    # Load configs
    with open("/root/configs/models.yaml")    as f: models_cfg    = yaml.safe_load(f)["generators"]
    with open("/root/configs/retrievers.yaml") as f: retrievers_cfg = yaml.safe_load(f)["retrievers"]
    with open("/root/configs/scaffolds.yaml")  as f: scaffolds_cfg  = yaml.safe_load(f)["scaffolds"]
    with open("/root/configs/experiment_phase_1_1.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    # Load questions
    subset_path = f"/root/{exp_cfg['question_subset']}"
    with open(subset_path) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    model_cfg    = models_cfg[generator]
    retr_cfg     = retrievers_cfg[retriever]
    scaffold_cfg = scaffolds_cfg[scaffold]
    top_k        = scaffold_cfg["top_k"]

    # Build retriever and generator
    if retr_cfg["type"] == "dense":
        ret = DenseRetriever()
    else:
        ret = HybridRetriever(
            alpha_dense=retr_cfg.get("alpha_dense", 0.5),
            alpha_bm25=retr_cfg.get("alpha_bm25", 0.5)
        )
    gen = AnthropicGenerator(model_cfg["model_name"])

    # Run
    records = []
    for q in tqdm(questions, desc=sid):
        retrieved = ret.search(q["question"], top_k=top_k)
        prompt    = build_prompt(q["question"], retrieved)
        try:
            answer = gen.generate(prompt)
            status = "success"
            error  = None
        except Exception as e:
            answer = ""
            status = "error"
            error  = str(e)

        records.append({
            "question_id"     : q["question_id"],
            "question"        : q["question"],
            "question_type"   : q["question_type"],
            "source"          : q["source"],
            "gold_doc_id"     : q["gold_doc_id"],
            "gold_section_id" : q["gold_section_id"],
            "reference_answer": q["reference_answer"],
            "system_id"       : sid,
            "generator_key"   : generator,
            "generator_model" : model_cfg["model_name"],
            "retriever_key"   : retriever,
            "retriever_type"  : retr_cfg["type"],
            "scaffold_key"    : scaffold,
            "top_k"           : top_k,
            "retrieved_chunks": retrieved,
            "generated_answer": answer,
            "run_status"      : status,
            "error"           : error,
        })

    # Save to volume
    out_path = f"/root/outputs/{sid}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    volume.commit()
    print(f"[modal] Done: {sid} — {len(records)} records saved to {out_path}")
    return {"system_id": sid, "n_records": len(records), "status": "success"}


# ── CORE RUN FUNCTION (GPU — Qwen local_open) ─────────────────────────────────
@app.function(
    image   = gpu_image,
    gpu     = "T4",
    timeout = 3600,
    volumes = {"/root/outputs": volume},
)
def run_qwen_system(generator: str, retriever: str, scaffold: str, limit: int = None) -> dict:
    import sys
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    sys.path.insert(0, "/root")

    from src.rag.retrieve import DenseRetriever, HybridRetriever
    from src.rag.generate import HFLocalGenerator, build_prompt
    import yaml
    import json
    from tqdm import tqdm

    sid = f"{generator}__{retriever}__{scaffold}"
    print(f"[modal] Starting: {sid}")

    with open("/root/configs/models.yaml")    as f: models_cfg    = yaml.safe_load(f)["generators"]
    with open("/root/configs/retrievers.yaml") as f: retrievers_cfg = yaml.safe_load(f)["retrievers"]
    with open("/root/configs/scaffolds.yaml")  as f: scaffolds_cfg  = yaml.safe_load(f)["scaffolds"]
    with open("/root/configs/experiment_phase_1_1.yaml") as f:
        exp_cfg = yaml.safe_load(f)

    subset_path = f"/root/{exp_cfg['question_subset']}"
    with open(subset_path) as f:
        questions = json.load(f)
    if limit:
        questions = questions[:limit]

    model_cfg    = models_cfg[generator]
    retr_cfg     = retrievers_cfg[retriever]
    scaffold_cfg = scaffolds_cfg[scaffold]
    top_k        = scaffold_cfg["top_k"]

    if retr_cfg["type"] == "dense":
        ret = DenseRetriever()
    else:
        ret = HybridRetriever(
            alpha_dense=retr_cfg.get("alpha_dense", 0.5),
            alpha_bm25=retr_cfg.get("alpha_bm25", 0.5)
        )
    gen = HFLocalGenerator(model_cfg["model_name"])

    records = []
    for q in tqdm(questions, desc=sid):
        retrieved = ret.search(q["question"], top_k=top_k)
        prompt    = build_prompt(q["question"], retrieved)
        try:
            answer = gen.generate(prompt)
            status = "success"
            error  = None
        except Exception as e:
            answer = ""
            status = "error"
            error  = str(e)

        records.append({
            "question_id"     : q["question_id"],
            "question"        : q["question"],
            "question_type"   : q["question_type"],
            "source"          : q["source"],
            "gold_doc_id"     : q["gold_doc_id"],
            "gold_section_id" : q["gold_section_id"],
            "reference_answer": q["reference_answer"],
            "system_id"       : sid,
            "generator_key"   : generator,
            "generator_model" : model_cfg["model_name"],
            "retriever_key"   : retriever,
            "retriever_type"  : retr_cfg["type"],
            "scaffold_key"    : scaffold,
            "top_k"           : top_k,
            "retrieved_chunks": retrieved,
            "generated_answer": answer,
            "run_status"      : status,
            "error"           : error,
        })

    out_path = f"/root/outputs/{sid}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    volume.commit()
    print(f"[modal] Done: {sid} — {len(records)} records saved")
    return {"system_id": sid, "n_records": len(records), "status": "success"}


# ── LOCAL ENTRYPOINT ──────────────────────────────────────────────────────────
@app.local_entrypoint()
def main(limit: int = None, system: str = None):
    """
    Run all 12 systems in parallel on Modal.

    Args:
        limit  : number of questions per system (None = all 100)
        system : run only this system ID (None = all 12)
    """
    print("=" * 60)
    print("  CS321M RAG EVAL — Modal Parallel Runner")
    print("=" * 60)

    if limit:
        print(f"  TEST MODE: {limit} questions per system")
    else:
        print(f"  FULL MODE: 100 questions per system")

    # Filter systems if specific one requested
    systems_to_run = ALL_SYSTEMS
    if system:
        systems_to_run = [
            s for s in ALL_SYSTEMS
            if system_id(s) == system
        ]
        if not systems_to_run:
            print(f"  ERROR: System '{system}' not found")
            return

    print(f"  Systems to run: {len(systems_to_run)}")
    print()

    # Separate Claude and Qwen systems
    claude_systems = [s for s in systems_to_run if s["generator"] != "local_open"]
    qwen_systems   = [s for s in systems_to_run if s["generator"] == "local_open"]

    print(f"  Claude systems (CPU): {len(claude_systems)}")
    print(f"  Qwen systems (GPU T4): {len(qwen_systems)}")
    print()

    # Launch all in parallel using .map()
    all_results = []

    if claude_systems:
        print("Launching Claude systems in parallel...")
        claude_results = list(run_claude_system.starmap([
            (s["generator"], s["retriever"], s["scaffold"], limit)
            for s in claude_systems
        ]))
        all_results.extend(claude_results)

    if qwen_systems:
        print("Launching Qwen systems in parallel...")
        qwen_results = list(run_qwen_system.starmap([
            (s["generator"], s["retriever"], s["scaffold"], limit)
            for s in qwen_systems
        ]))
        all_results.extend(qwen_results)

    # Summary
    print()
    print("=" * 60)
    print("  COMPLETE")
    print("=" * 60)
    for r in all_results:
        status = "✅" if r["status"] == "success" else "❌"
        print(f"  {status} {r['system_id']} — {r['n_records']} records")

    print()
    print("Download results with:")
    print("  bash scripts/download_modal_results.sh")
    print("=" * 60)
