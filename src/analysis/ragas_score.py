"""
ragas_score.py
===============
RAG Evaluation Scoring — Faithfulness + Answer Relevance
RAGAS 0.1.21 | Claude Haiku judge | MiniLM embeddings

Run in cs321m_ragas environment (Python 3.11):
  conda activate cs321m_ragas
  python src/analysis/ragas_score.py --limit 3   # test
  python src/analysis/ragas_score.py              # full run

Replaces ragas_eval.py (mock version). This is the real scoring script.

Input  : outputs/runs/phase_1_1/*.jsonl (12 system run files, 100 records each)
Output : outputs/phase1/ragas/
  per_query_scores.json   — 1200 records with faithfulness + answer_relevance
  ragas_report.json       — aggregate metrics per system
  ragas_report.txt        — human-readable report
  ragas_by_system.png     — scores per system
  ragas_pass_rates.png    — pass rates per system (input to Glicko-2)
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "judge_model"      : "claude-haiku-4-5",
    "embedder_model"   : "sentence-transformers/all-MiniLM-L6-v2",
    "pass_threshold"   : 0.75,
    "good_threshold"   : 0.75,
    "warning_threshold": 0.55,
    "batch_size"       : 10,
    "system_ids": [
        "strong_api__dense__topk3",
        "strong_api__dense__topk8",
        "strong_api__hybrid__topk3",
        "strong_api__hybrid__topk8",
        "mid_api__dense__topk3",
        "mid_api__dense__topk8",
        "mid_api__hybrid__topk3",
        "mid_api__hybrid__topk8",
        "local_open__dense__topk3",
        "local_open__dense__topk8",
        "local_open__hybrid__topk3",
        "local_open__hybrid__topk8",
    ],
}

SYSTEM_LABELS = {
    "strong_api": "Claude Opus 4.5",
    "mid_api"   : "Claude Haiku 4.5",
    "local_open": "Qwen 2.5-3B",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD RUNS
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(runs_dir: str, limit: int = None) -> list:
    runs_path   = Path(runs_dir)
    all_records = []

    print(f"\n[1/4] Loading run files from {runs_dir}")

    for system_id in CONFIG["system_ids"]:
        path = runs_path / f"{system_id}.jsonl"
        if not path.exists():
            print(f"      MISSING: {system_id}.jsonl — skipping")
            continue

        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

        if limit:
            records = records[:limit]

        successful = [
            r for r in records
            if r.get("run_status") == "success"
            and r.get("generated_answer", "").strip()
        ]
        skipped = len(records) - len(successful)

        print(f"      {system_id}: {len(successful)} records"
              f"{f' ({skipped} skipped)' if skipped else ''}")

        all_records.extend(successful)

    print(f"\n      Total records to score: {len(all_records)}")
    return all_records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: RAGAS SETUP (0.1.21 API)
# ─────────────────────────────────────────────────────────────────────────────

def setup_ragas():
    """
    RAGAS 0.1.21 setup.
    - LLM judge  : Claude Haiku via langchain-anthropic + LangchainLLMWrapper
    - Embeddings : all-MiniLM-L6-v2 via sentence-transformers (local, free, fast)
    - Metrics    : faithfulness (LLM-only) + answer_relevancy (LLM + embeddings)
    - Input      : HuggingFace Dataset with columns:
                   question, contexts (list of str), answer, ground_truth
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_anthropic import ChatAnthropic
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # LLM judge
    llm       = ChatAnthropic(model=CONFIG["judge_model"], temperature=0)
    ragas_llm = LangchainLLMWrapper(llm)

    # Embeddings (local, no API cost)
    embedder       = HuggingFaceEmbeddings(model_name=CONFIG["embedder_model"])
    ragas_embedder = LangchainEmbeddingsWrapper(embedder)

    # Inject into metrics
    faithfulness.llm       = ragas_llm
    answer_relevancy.llm   = ragas_llm
    answer_relevancy.embeddings = ragas_embedder

    print(f"      LLM judge : {CONFIG['judge_model']}")
    print(f"      Embedder  : {CONFIG['embedder_model']} (local)")

    return evaluate, [faithfulness, answer_relevancy]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SCORE RECORDS
# ─────────────────────────────────────────────────────────────────────────────

def score_batch(records: list, evaluate_fn, metrics) -> list:
    from datasets import Dataset

    rows = []
    for r in records:
        contexts = [
            chunk.get("text", "")
            for chunk in r.get("retrieved_chunks", [])
        ]
        rows.append({
            "question"    : r["question"],
            "contexts"    : contexts,
            "answer"      : r["generated_answer"],
            "ground_truth": r.get("reference_answer", ""),
        })

    dataset = Dataset.from_list(rows)
    result  = evaluate_fn(dataset, metrics=metrics)
    df      = result.to_pandas()

    scored = []
    for i, r in enumerate(records):
        faith = float(df.iloc[i].get("faithfulness", 0.0) or 0.0)
        rel   = float(df.iloc[i].get("answer_relevancy", 0.0) or 0.0)
        comp  = round((faith + rel) / 2, 4)
        scored.append({
            **r,
            "faithfulness"     : round(faith, 4),
            "answer_relevance" : round(rel, 4),
            "composite_ragas"  : comp,
            "pass_fail"        : int(comp >= CONFIG["pass_threshold"]),
            "ragas_judge_model": CONFIG["judge_model"],
        })
    return scored


def score_all_records(all_records: list, output_dir: str) -> list:
    batch_size      = CONFIG["batch_size"]
    checkpoint_path = Path(output_dir) / "per_query_scores.json"

    print(f"\n[2/4] Scoring {len(all_records)} records")
    print(f"      Judge     : {CONFIG['judge_model']}")
    print(f"      Batch size: {batch_size}")

    # Load checkpoint
    scored_ids = set()
    all_scored = []
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            try:
                all_scored = json.load(f)
                scored_ids = {
                    r["question_id"] + r["system_id"]
                    for r in all_scored
                    if "faithfulness" in r
                }
                print(f"      Resuming — {len(all_scored)} already scored")
            except Exception:
                all_scored = []

    to_score = [
        r for r in all_records
        if r["question_id"] + r["system_id"] not in scored_ids
    ]
    print(f"      Remaining : {len(to_score)} records\n")

    if not to_score:
        print("      All records already scored.")
        return all_scored

    # Setup RAGAS once
    evaluate_fn, metrics = setup_ragas()
    n_batches = (len(to_score) + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end   = min(start + batch_size, len(to_score))
        batch = to_score[start:end]
        systems = set(r["system_id"] for r in batch)

        print(f"  Batch {batch_idx+1}/{n_batches} "
              f"(records {start+1}–{end}) | "
              f"{', '.join(systems)}")

        try:
            scored_batch = score_batch(batch, evaluate_fn, metrics)
            all_scored.extend(scored_batch)

            for r in scored_batch:
                print(f"    {r['system_id']:<35} "
                      f"faith={r['faithfulness']:.3f} "
                      f"rel={r['answer_relevance']:.3f} "
                      f"pass={r['pass_fail']}")

            with open(checkpoint_path, "w") as f:
                json.dump(all_scored, f, indent=2)
            print(f"    Checkpoint saved ({len(all_scored)} total)\n")

        except Exception as e:
            print(f"    ⚠ Batch {batch_idx+1} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n      Scoring complete. {len(all_scored)} records scored.")
    return all_scored


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: METRICS + VISUALIZATIONS + SAVE
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(scored_records: list) -> dict:
    print(f"\n[3/4] Computing aggregate metrics...")

    system_stats = defaultdict(lambda: {
        "faithfulness": [], "answer_relevance": [],
        "composite_ragas": [], "pass_fail": []
    })

    for r in scored_records:
        sid = r["system_id"]
        system_stats[sid]["faithfulness"].append(r.get("faithfulness", 0))
        system_stats[sid]["answer_relevance"].append(r.get("answer_relevance", 0))
        system_stats[sid]["composite_ragas"].append(r.get("composite_ragas", 0))
        system_stats[sid]["pass_fail"].append(r.get("pass_fail", 0))

    system_summary = {}
    for sid, stats in system_stats.items():
        gen_key = sid.split("__")[0]
        system_summary[sid] = {
            "system_id"            : sid,
            "generator"            : gen_key,
            "generator_label"      : SYSTEM_LABELS.get(gen_key, gen_key),
            "n_records"            : len(stats["faithfulness"]),
            "faithfulness_mean"    : round(np.mean(stats["faithfulness"]), 4),
            "faithfulness_std"     : round(np.std(stats["faithfulness"]), 4),
            "answer_relevance_mean": round(np.mean(stats["answer_relevance"]), 4),
            "answer_relevance_std" : round(np.std(stats["answer_relevance"]), 4),
            "composite_mean"       : round(np.mean(stats["composite_ragas"]), 4),
            "composite_std"        : round(np.std(stats["composite_ragas"]), 4),
            "pass_rate"            : round(np.mean(stats["pass_fail"]), 4),
        }
        s = system_summary[sid]
        print(f"      {sid:<35} "
              f"faith={s['faithfulness_mean']:.3f} "
              f"rel={s['answer_relevance_mean']:.3f} "
              f"pass={s['pass_rate']:.3f}")

    all_faith = [r.get("faithfulness", 0)     for r in scored_records]
    all_rel   = [r.get("answer_relevance", 0) for r in scored_records]
    all_comp  = [r.get("composite_ragas", 0)  for r in scored_records]
    all_pass  = [r.get("pass_fail", 0)        for r in scored_records]

    overall = {
        "n_total"              : len(scored_records),
        "n_systems"            : len(system_stats),
        "faithfulness_mean"    : round(np.mean(all_faith), 4),
        "answer_relevance_mean": round(np.mean(all_rel), 4),
        "composite_mean"       : round(np.mean(all_comp), 4),
        "overall_pass_rate"    : round(np.mean(all_pass), 4),
    }

    return {"overall": overall, "system_summary": system_summary}


def generate_visualizations(scored_records: list, metrics: dict, output_dir: str):
    print(f"\n[4/4] Generating visualizations...")
    out = Path(output_dir)
    ss  = metrics["system_summary"]
    ov  = metrics["overall"]

    systems = sorted(ss.keys(), key=lambda s: ss[s]["composite_mean"], reverse=True)
    gen_colors = {
        "strong_api": "#4CAF50",
        "mid_api"   : "#2196F3",
        "local_open": "#FF9800",
    }

    # Plot 1: Faithfulness + Answer Relevance
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, label in [
        (axes[0], "faithfulness_mean",     "Faithfulness"),
        (axes[1], "answer_relevance_mean", "Answer Relevance"),
    ]:
        values = [ss[s][metric] for s in systems]
        colors = [gen_colors.get(s.split("__")[0], "gray") for s in systems]
        bars   = ax.barh([s.replace("__", "\n") for s in systems],
                         values, color=colors, edgecolor="white", alpha=0.85)
        ax.axvline(x=CONFIG["good_threshold"], color="green",
                   linestyle="--", linewidth=1.5, label=f"Good ({CONFIG['good_threshold']})")
        ax.axvline(x=CONFIG["warning_threshold"], color="orange",
                   linestyle="--", linewidth=1.2, label=f"Warning ({CONFIG['warning_threshold']})")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_title(f"{label} by System", fontsize=12)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    from matplotlib.patches import Patch
    fig.legend(handles=[
        Patch(facecolor="#4CAF50", label="Claude Opus 4.5 (strong_api)"),
        Patch(facecolor="#2196F3", label="Claude Haiku 4.5 (mid_api)"),
        Patch(facecolor="#FF9800", label="Qwen 2.5-3B (local_open)"),
    ], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(
        f"RAG Evaluation by System | "
        f"faith={ov['faithfulness_mean']:.3f} | rel={ov['answer_relevance_mean']:.3f}",
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(out / "ragas_by_system.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: ragas_by_system.png")

    # Plot 2: Pass rates
    fig, ax = plt.subplots(figsize=(13, 5))
    pass_rates = [ss[s]["pass_rate"] for s in systems]
    colors     = [gen_colors.get(s.split("__")[0], "gray") for s in systems]
    bars       = ax.bar(range(len(systems)), pass_rates,
                        color=colors, edgecolor="white", alpha=0.85)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([s.replace("__", "\n") for s in systems], fontsize=8)
    ax.axhline(y=ov["overall_pass_rate"], color="black", linestyle="--",
               linewidth=1.5, label=f"Overall ({ov['overall_pass_rate']:.3f})")
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Pass Rate by System (threshold={CONFIG['pass_threshold']}) — Input to Glicko-2")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ragas_pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: ragas_pass_rates.png")


def save_outputs(scored_records: list, metrics: dict, output_dir: str):
    out = Path(output_dir)
    ov  = metrics["overall"]
    ss  = metrics["system_summary"]

    with open(out / "ragas_report.json", "w") as f:
        json.dump({
            "generated_at"  : datetime.utcnow().isoformat(),
            "judge_model"   : CONFIG["judge_model"],
            "embedder_model": CONFIG["embedder_model"],
            "ragas_version" : "0.1.21",
            "pass_threshold": CONFIG["pass_threshold"],
            "overall"       : ov,
            "system_summary": ss,
        }, f, indent=2)

    sorted_systems = sorted(ss.values(), key=lambda x: x["composite_mean"], reverse=True)
    lines = [
        "=" * 65,
        "  RAG EVALUATION REPORT — Stanford CS321M Phase 1",
        f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Judge     : {CONFIG['judge_model']} via RAGAS 0.1.21",
        f"  Threshold : {CONFIG['pass_threshold']} (pass/fail for Glicko-2)",
        "=" * 65,
        "",
        "OVERALL",
        "-" * 40,
        f"  Records scored       : {ov['n_total']}",
        f"  Systems evaluated    : {ov['n_systems']}",
        f"  Faithfulness mean    : {ov['faithfulness_mean']:.4f}",
        f"  Answer relevance mean: {ov['answer_relevance_mean']:.4f}",
        f"  Composite mean       : {ov['composite_mean']:.4f}",
        f"  Overall pass rate    : {ov['overall_pass_rate']:.4f}",
        "",
        "PER-SYSTEM (sorted by composite)",
        "-" * 40,
        f"  {'System':<35} {'Faith':>7} {'Rel':>7} {'Comp':>7} {'Pass':>7}",
        f"  {'-'*59}",
    ]
    for s in sorted_systems:
        lines.append(
            f"  {s['system_id']:<35} "
            f"{s['faithfulness_mean']:>7.4f} "
            f"{s['answer_relevance_mean']:>7.4f} "
            f"{s['composite_mean']:>7.4f} "
            f"{s['pass_rate']:>7.4f}"
        )
    lines += ["", "NOTE: pass_fail feeds into Phase 2 Glicko-2.", "=" * 65]

    with open(out / "ragas_report.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\n      All outputs saved to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(runs_dir: str, output_dir: str, limit: int = None):
    print("\n" + "=" * 65)
    print("  RAGAS SCORING — Stanford CS321M Phase 1")
    print("=" * 65)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_records = load_runs(runs_dir, limit=limit)
    if not all_records:
        print("No records found.")
        return

    scored_records = score_all_records(all_records, output_dir)
    metrics        = compute_metrics(scored_records)
    generate_visualizations(scored_records, metrics, output_dir)
    save_outputs(scored_records, metrics, output_dir)

    ov = metrics["overall"]
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Records scored   : {ov['n_total']}")
    print(f"  Faithfulness     : {ov['faithfulness_mean']:.4f}")
    print(f"  Answer Relevance : {ov['answer_relevance_mean']:.4f}")
    print(f"  Composite        : {ov['composite_mean']:.4f}")
    print(f"  Overall pass rate: {ov['overall_pass_rate']:.4f}")
    print()
    print("  Top 3 systems:")
    for s in sorted(metrics["system_summary"].values(),
                    key=lambda x: x["composite_mean"], reverse=True)[:3]:
        print(f"    {s['system_id']:<35} composite={s['composite_mean']:.4f}")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",   default="outputs/runs/phase_1_1")
    parser.add_argument("--output_dir", default="outputs/phase1/ragas")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.runs_dir, args.output_dir, args.limit)
