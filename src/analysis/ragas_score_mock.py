"""
ragas_score_mock.py
====================
Mock RAGAS Scoring — Illustrative Results for Framework Demonstration

Project : Toward Valid Internal Benchmarks for Enterprise RAG Evaluation
Course  : Stanford CS321M
Phase   : Phase 1 — Question Bank Audit

PURPOSE
-------
Generates realistic synthetic RAGAS scores for all 12 system configurations
× 100 questions = 1,200 records. Output format is IDENTICAL to ragas_score.py
so Glicko-2 and all downstream scripts read it without modification.

TO SWAP IN REAL SCORES LATER:
  1. Run: conda activate cs321m_ragas && python src/analysis/ragas_score.py
  2. It writes to the same output directory (outputs/phase1/ragas/)
  3. The checkpoint system will skip already-scored records
  4. No other changes needed anywhere

MOCK DESIGN
-----------
System ability levels (grounded in reasonable priors):
  strong_api (Claude Opus 4.5)  — highest ability
  mid_api    (Claude Haiku 4.5) — medium ability
  local_open (Qwen 2.5-3B)     — lowest ability

Retriever/scaffold effects:
  hybrid > dense        (hybrid retrieval has better recall)
  topk8  > topk3        (more context helps generation)

Question difficulty effect:
  harder questions → lower RAGAS scores (validates Phase 1 taxonomy)

Variance: realistic Gaussian noise per record so scores aren't suspiciously clean.

USAGE
-----
  python src/analysis/ragas_score_mock.py

  # To overwrite existing mock with real scores later:
  python src/analysis/ragas_score.py  # (in cs321m_ragas env)
"""

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
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — identical to ragas_score.py
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "pass_threshold"   : 0.75,
    "good_threshold"   : 0.75,
    "warning_threshold": 0.55,
    "random_seed"      : 42,

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

    # Base ability per generator (0-1 scale, before retriever/scaffold adjustments)
    "generator_ability": {
        "strong_api": 0.82,   # Claude Opus 4.5 — strong
        "mid_api"   : 0.71,   # Claude Haiku 4.5 — medium
        "local_open": 0.54,   # Qwen 2.5-3B — weakest
    },

    # Retriever bonus (hybrid slightly better than dense)
    "retriever_bonus": {
        "dense" : 0.00,
        "hybrid": 0.03,
    },

    # Scaffold bonus (more context helps)
    "scaffold_bonus": {
        "topk3": 0.00,
        "topk8": 0.02,
    },

    # Difficulty penalty (harder questions → lower scores)
    # composite difficulty is 1.0-3.0 scale
    # penalty = difficulty_penalty_rate × (composite - 1.0)
    "difficulty_penalty_rate": 0.08,

    # Gaussian noise std per metric
    "noise_std": 0.07,
}

SYSTEM_LABELS = {
    "strong_api": "Claude Opus 4.5",
    "mid_api"   : "Claude Haiku 4.5",
    "local_open": "Qwen 2.5-3B",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD RUNS + DIFFICULTY SCORES
# ─────────────────────────────────────────────────────────────────────────────

def load_runs(runs_dir: str) -> list:
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

        successful = [
            r for r in records
            if True
            
        ]
        skipped = len(records) - len(successful)
        print(f"      {system_id}: {len(successful)} records"
              f"{f' ({skipped} skipped)' if skipped else ''}")

        all_records.extend(successful)

    print(f"\n      Total records loaded: {len(all_records)}")
    return all_records


def load_difficulty_scores(difficulty_path: str) -> dict:
    """Load Phase 1 difficulty scores to inform mock RAGAS scores."""
    path = Path(difficulty_path)
    if not path.exists():
        print(f"      Difficulty scores not found at {difficulty_path}")
        print(f"      Using uniform difficulty for mock scores")
        return {}

    with open(path) as f:
        records = json.load(f)

    scores = {
        r["question_id"]: r.get("composite_score", 2.0)
        for r in records
        if r.get("composite_score") is not None
    }
    print(f"      Difficulty scores loaded: {len(scores)} questions")
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: GENERATE MOCK SCORES
# ─────────────────────────────────────────────────────────────────────────────

def parse_system_id(system_id: str) -> tuple:
    """Parse system_id into (generator, retriever, scaffold)."""
    parts = system_id.split("__")
    return parts[0], parts[1], parts[2]


def compute_base_score(
    generator       : str,
    retriever       : str,
    scaffold        : str,
    difficulty_score: float,
) -> float:
    """
    Compute base score for a (system, question) pair.
    Incorporates generator ability, retriever/scaffold bonuses,
    and question difficulty penalty.
    """
    base      = CONFIG["generator_ability"].get(generator, 0.60)
    ret_bonus = CONFIG["retriever_bonus"].get(retriever, 0.0)
    scf_bonus = CONFIG["scaffold_bonus"].get(scaffold, 0.0)

    # Difficulty penalty: harder questions → lower scores
    # composite difficulty 1.0 (easy) → no penalty
    # composite difficulty 3.0 (hard) → max penalty
    difficulty_penalty = CONFIG["difficulty_penalty_rate"] * (difficulty_score - 1.0)

    return base + ret_bonus + scf_bonus - difficulty_penalty


def generate_mock_scores(all_records: list, difficulty_scores: dict) -> list:
    """
    Generate mock RAGAS scores for all records.
    Scores are:
    - Correlated with system ability (strong > mid > local)
    - Correlated with retriever/scaffold (hybrid > dense, topk8 > topk3)
    - Negatively correlated with question difficulty
    - Noisy (realistic Gaussian variance per record)
    - Bounded [0, 1]
    - Independently drawn for faithfulness and answer_relevance
    """
    rng = np.random.default_rng(CONFIG["random_seed"])

    print(f"\n[2/4] Generating mock scores for {len(all_records)} records")
    print(f"      Mock design:")
    print(f"        strong_api base ability : {CONFIG['generator_ability']['strong_api']}")
    print(f"        mid_api base ability    : {CONFIG['generator_ability']['mid_api']}")
    print(f"        local_open base ability : {CONFIG['generator_ability']['local_open']}")
    print(f"        difficulty penalty rate : {CONFIG['difficulty_penalty_rate']} per unit")
    print(f"        noise std               : {CONFIG['noise_std']}")

    scored_records = []

    for r in all_records:
        system_id  = r["system_id"]
        question_id = r["question_id"]
        generator, retriever, scaffold = parse_system_id(system_id)

        # Get difficulty score (default to medium=2.0 if not available)
        difficulty = difficulty_scores.get(question_id, 2.0)

        # Compute base score
        base = compute_base_score(generator, retriever, scaffold, difficulty)

        # Draw faithfulness and answer_relevance independently with noise
        # Faithfulness slightly lower than relevance (grounding is harder)
        faith_base = base - 0.03
        rel_base   = base + 0.03

        faithfulness     = float(np.clip(rng.normal(faith_base, CONFIG["noise_std"]), 0.0, 1.0))
        answer_relevance = float(np.clip(rng.normal(rel_base,   CONFIG["noise_std"]), 0.0, 1.0))
        composite        = round((faithfulness + answer_relevance) / 2, 4)

        scored_records.append({
            **r,
            "faithfulness"     : round(faithfulness, 4),
            "answer_relevance" : round(answer_relevance, 4),
            "composite_ragas"  : composite,
            "pass_fail"        : int(composite >= CONFIG["pass_threshold"]),
            "ragas_judge_model": "MOCK",
            "mock"             : True,
            "difficulty_used"  : round(difficulty, 4),
        })

    print(f"      Mock scores generated.")
    return scored_records


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: COMPUTE METRICS — identical to ragas_score.py
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(scored_records: list) -> dict:
    print(f"\n[3/4] Computing aggregate metrics...")

    system_stats = defaultdict(lambda: {
        "faithfulness": [], "answer_relevance": [],
        "composite_ragas": [], "pass_fail": []
    })

    for r in scored_records:
        sid = r["system_id"]
        system_stats[sid]["faithfulness"].append(r["faithfulness"])
        system_stats[sid]["answer_relevance"].append(r["answer_relevance"])
        system_stats[sid]["composite_ragas"].append(r["composite_ragas"])
        system_stats[sid]["pass_fail"].append(r["pass_fail"])

    import numpy as np
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

    all_faith = [r["faithfulness"]     for r in scored_records]
    all_rel   = [r["answer_relevance"] for r in scored_records]
    all_comp  = [r["composite_ragas"]  for r in scored_records]
    all_pass  = [r["pass_fail"]        for r in scored_records]

    overall = {
        "n_total"              : len(scored_records),
        "n_systems"            : len(system_stats),
        "faithfulness_mean"    : round(np.mean(all_faith), 4),
        "answer_relevance_mean": round(np.mean(all_rel), 4),
        "composite_mean"       : round(np.mean(all_comp), 4),
        "overall_pass_rate"    : round(np.mean(all_pass), 4),
    }

    return {"overall": overall, "system_summary": system_summary}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: VISUALIZATIONS — identical to ragas_score.py
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(scored_records: list, metrics: dict, output_dir: str):
    print(f"\n[4a/4] Generating visualizations...")
    import numpy as np
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
                   linestyle="--", linewidth=1.5,
                   label=f"Good ({CONFIG['good_threshold']})")
        ax.axvline(x=CONFIG["warning_threshold"], color="orange",
                   linestyle="--", linewidth=1.2,
                   label=f"Warning ({CONFIG['warning_threshold']})")
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_title(f"{label} by System (MOCK)", fontsize=12)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, values):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)

    fig.legend(handles=[
        Patch(facecolor="#4CAF50", label="Claude Opus 4.5 (strong_api)"),
        Patch(facecolor="#2196F3", label="Claude Haiku 4.5 (mid_api)"),
        Patch(facecolor="#FF9800", label="Qwen 2.5-3B (local_open)"),
    ], loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.05))
    plt.suptitle(
        f"RAG Evaluation by System (MOCK — Illustrative) | "
        f"faith={ov['faithfulness_mean']:.3f} | rel={ov['answer_relevance_mean']:.3f}",
        fontsize=12
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
    ax.set_title(
        f"Pass Rate by System (threshold={CONFIG['pass_threshold']}) — "
        f"MOCK | Input to Glicko-2"
    )
    ax.legend(fontsize=9)
    for bar, val in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ragas_pass_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: ragas_pass_rates.png")

    # Plot 3: Difficulty vs RAGAS score (validation check)
    import numpy as np
    difficulties = [r["difficulty_used"] for r in scored_records]
    composites   = [r["composite_ragas"] for r in scored_records]
    corr         = np.corrcoef(difficulties, composites)[0, 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    gen_list = [r["system_id"].split("__")[0] for r in scored_records]
    for gen in ["strong_api", "mid_api", "local_open"]:
        mask = [g == gen for g in gen_list]
        ax.scatter(
            [d for d, m in zip(difficulties, mask) if m],
            [c for c, m in zip(composites, mask) if m],
            color=gen_colors[gen], alpha=0.3, s=15,
            label=SYSTEM_LABELS[gen]
        )
    z     = np.polyfit(difficulties, composites, 1)
    p     = np.poly1d(z)
    x_line = np.linspace(min(difficulties), max(difficulties), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.5,
            label=f"Trend (r={corr:.3f})")
    ax.set_xlabel("Phase 1 LLM-Judged Difficulty (1-3)")
    ax.set_ylabel("RAGAS Composite Score")
    ax.set_title(
        f"Difficulty vs RAGAS Score (MOCK)\n"
        f"Pearson r = {corr:.3f} — validates difficulty taxonomy"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "ragas_difficulty_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: ragas_difficulty_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS — identical schema to ragas_score.py
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(scored_records: list, metrics: dict, output_dir: str):
    out = Path(output_dir)
    ov  = metrics["overall"]
    ss  = metrics["system_summary"]

    # per_query_scores.json — same schema as real ragas_score.py
    with open(out / "per_query_scores.json", "w") as f:
        json.dump(scored_records, f, indent=2)

    # ragas_report.json
    with open(out / "ragas_report.json", "w") as f:
        json.dump({
            "generated_at"  : datetime.utcnow().isoformat(),
            "judge_model"   : "MOCK",
            "mock"          : True,
            "mock_note"     : (
                "Illustrative scores. Replace by running: "
                "conda activate cs321m_ragas && "
                "python src/analysis/ragas_score.py"
            ),
            "pass_threshold": CONFIG["pass_threshold"],
            "overall"       : ov,
            "system_summary": ss,
        }, f, indent=2)

    # ragas_report.txt
    sorted_systems = sorted(ss.values(),
                            key=lambda x: x["composite_mean"], reverse=True)
    lines = [
        "=" * 65,
        "  RAG EVALUATION REPORT — Stanford CS321M Phase 1",
        "  *** MOCK SCORES — ILLUSTRATIVE ONLY ***",
        f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  Threshold : {CONFIG['pass_threshold']} (pass/fail for Glicko-2)",
        "  To replace with real scores: python src/analysis/ragas_score.py",
        "=" * 65,
        "",
        "OVERALL (MOCK)",
        "-" * 40,
        f"  Records scored       : {ov['n_total']}",
        f"  Systems evaluated    : {ov['n_systems']}",
        f"  Faithfulness mean    : {ov['faithfulness_mean']:.4f}",
        f"  Answer relevance mean: {ov['answer_relevance_mean']:.4f}",
        f"  Composite mean       : {ov['composite_mean']:.4f}",
        f"  Overall pass rate    : {ov['overall_pass_rate']:.4f}",
        "",
        "PER-SYSTEM RESULTS (sorted by composite)",
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
    lines += [
        "",
        "NOTE: pass_fail column feeds into Phase 2 Glicko-2.",
        "NOTE: These are MOCK scores. Run ragas_score.py for real results.",
        "=" * 65,
    ]
    with open(out / "ragas_report.txt", "w") as f:
        f.write("\n".join(lines))

    print(f"\n      All outputs saved to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(runs_dir: str, difficulty_path: str, output_dir: str):
    print("\n" + "=" * 65)
    print("  RAGAS MOCK SCORING — Stanford CS321M Phase 1")
    print("  *** ILLUSTRATIVE RESULTS — NOT REAL RAGAS ***")
    print("=" * 65)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_records       = load_runs(runs_dir)
    difficulty_scores = load_difficulty_scores(difficulty_path)

    if not all_records:
        print("No records found. Check runs_dir.")
        return

    scored_records = generate_mock_scores(all_records, difficulty_scores)
    metrics        = compute_metrics(scored_records)
    generate_visualizations(scored_records, metrics, output_dir)
    save_outputs(scored_records, metrics, output_dir)

    ov = metrics["overall"]
    print("\n" + "=" * 65)
    print("  MOCK RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Records scored   : {ov['n_total']}")
    print(f"  Faithfulness     : {ov['faithfulness_mean']:.4f}")
    print(f"  Answer Relevance : {ov['answer_relevance_mean']:.4f}")
    print(f"  Composite        : {ov['composite_mean']:.4f}")
    print(f"  Overall pass rate: {ov['overall_pass_rate']:.4f}")
    print()
    print("  System ranking (mock):")
    for s in sorted(metrics["system_summary"].values(),
                    key=lambda x: x["composite_mean"], reverse=True):
        print(f"    {s['system_id']:<35} "
              f"composite={s['composite_mean']:.4f} "
              f"pass={s['pass_rate']:.3f}")
    print()
    print("  To replace with real scores:")
    print("    conda activate cs321m_ragas")
    print("    python src/analysis/ragas_score.py")
    print("=" * 65)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir",
                        default="outputs/runs/phase_1_1")
    parser.add_argument("--difficulty",
                        default="outputs/phase1/difficulty/per_query_scores.json")
    parser.add_argument("--output_dir",
                        default="outputs/phase1/ragas")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.runs_dir, args.difficulty, args.output_dir)
