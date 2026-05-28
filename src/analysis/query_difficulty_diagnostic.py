"""
query_difficulty_diagnostic.py
================================
Automated Query Difficulty Diagnostic for RAG Question Banks

Project : Toward Valid Internal Benchmarks for Enterprise RAG Evaluation
Course  : Stanford CS321M
Phase   : Phase 1 — Question Bank Audit

PURPOSE
-------
This script scores every question in a RAG question bank on three
difficulty dimensions using an LLM-as-judge approach, then produces
a diagnostic report showing the difficulty distribution of the bank.

The three dimensions (each scored 1/2/3):
  1. Query Formulation Difficulty
     How directly or indirectly is the query phrased relative to how
     the answer would appear in a source document?

  2. Reasoning Demand
     How many inference steps are required to produce the correct
     answer, assuming the evidence has already been retrieved?

  3. Answer Form Complexity
     How complex must the correct answer be in terms of its form
     and completeness?

Each dimension is scored independently via a separate LLM call with
its own focused prompt. This prevents dimension conflation.

TAXONOMY GROUNDING
------------------
These dimensions emerge from a literature-based construct development
process across 7 RAG evaluation papers:
  - GRADE (Lee et al., EMNLP 2025)
  - EnterpriseRAG-Bench (Sun et al., 2026)
  - RGB (Chen et al., AAAI 2024)
  - FRAMES (Krishna et al., ACL 2024)
  - MultiHop-RAG (Tang & Yang, 2024)
  - Adaptive-RAG (Jeong et al., NAACL 2024)
  - LiveRAG (Carmel et al., SIGIR 2025)

INPUT
-----
  --questions     : path to question bank JSON
  --prompts_dir   : directory containing the three prompt .txt files
  --llm_provider  : anthropic | openai
  --model         : model name (default: claude-sonnet-4-20250514 or gpt-4o)
  --output_dir    : directory for all outputs

OUTPUT
------
  per_query_scores.json       per-question scores + reasoning
  difficulty_report.json      aggregate metrics
  difficulty_report.txt       human-readable diagnostic report
  distribution_overall.png    composite score histogram
  distribution_by_dimension.png  per-dimension score bars
  distribution_by_type.png    extractive vs abstractive breakdown
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Scoring scale
    "score_min"  : 1,
    "score_max"  : 3,

    # Difficulty bands (based on composite score)
    "band_easy"   : (1.0, 1.67),
    "band_medium" : (1.67, 2.33),
    "band_hard"   : (2.33, 3.01),

    # Thresholds for recommendations
    "skew_threshold"        : 0.50,   # >50% in one band = skewed
    "hard_minimum"          : 0.15,   # <15% hard = insufficient hard questions
    "easy_maximum"          : 0.60,   # >60% easy = dominated by easy queries

    # LLM call settings
    "max_retries"  : 2,
    "retry_delay"  : 2.0,   # seconds between retries
    "temperature"  : 0.0,   # deterministic
    "max_tokens"   : 500,

    # Prompt files
    "prompt_files" : {
        "query_formulation_difficulty" : "query_formulation.txt",
        "reasoning_demand"             : "reasoning_demand.txt",
        "answer_form_complexity"       : "answer_form.txt",
    },

    # Default models
    "default_models" : {
        "anthropic" : "claude-sonnet-4-20250514",
        "openai"    : "gpt-4o",
    },
}

DIMENSION_LABELS = {
    "query_formulation_difficulty" : "Query Formulation",
    "reasoning_demand"             : "Reasoning Demand",
    "answer_form_complexity"       : "Answer Form Complexity",
}

SCORE_LABELS = {1: "Low", 2: "Medium", 3: "High"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(questions_path: str) -> list:
    """Load question bank from JSON file."""
    print(f"\n[1/5] Loading questions from {questions_path}")
    with open(questions_path) as f:
        questions = json.load(f)
    print(f"      Questions loaded: {len(questions)}")
    assert len(questions) > 0, "Question file is empty"
    return questions


def load_prompts(prompts_dir: str) -> dict:
    """
    Load all three prompt templates from the prompts directory.
    Each prompt file contains {QUERY} as the placeholder for the question.
    """
    print(f"      Loading prompts from {prompts_dir}/")
    prompts = {}
    for dim_key, filename in CONFIG["prompt_files"].items():
        path = os.path.join(prompts_dir, filename)
        assert os.path.exists(path), \
            f"Prompt file not found: {path}\nExpected: {filename}"
        with open(path) as f:
            prompts[dim_key] = f.read()
        print(f"        Loaded: {filename} → {dim_key}")
    return prompts


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LLM JUDGE
# ─────────────────────────────────────────────────────────────────────────────

def build_anthropic_client():
    """Build Anthropic client."""
    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        )


def build_openai_client():
    """Build OpenAI client."""
    try:
        import openai
        return openai.OpenAI()
    except ImportError:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        )


def call_llm(
    prompt    : str,
    provider  : str,
    model     : str,
    client
) -> str:
    """
    Make a single LLM API call and return the raw text response.
    Handles both Anthropic and OpenAI APIs.
    """
    if provider == "anthropic":
        response = client.messages.create(
            model      = model,
            max_tokens = CONFIG["max_tokens"],
            temperature= CONFIG["temperature"],
            messages   = [{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    elif provider == "openai":
        response = client.chat.completions.create(
            model      = model,
            max_tokens = CONFIG["max_tokens"],
            temperature= CONFIG["temperature"],
            messages   = [{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")


def parse_llm_response(raw: str, dimension: str) -> dict:
    """
    Parse the LLM JSON response into a structured dict.
    Handles common formatting issues (markdown fences, extra text).

    Returns dict with keys: dimension, score, reasoning
    Returns None if parsing fails after cleanup.
    """
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            l for l in lines
            if not l.strip().startswith("```")
        ).strip()

    # Try to find JSON block if model added preamble
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end   = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]

    try:
        parsed = json.loads(cleaned)
        score  = int(parsed.get("score", -1))
        assert score in [1, 2, 3], f"Score {score} not in [1,2,3]"
        return {
            "dimension" : dimension,
            "score"     : score,
            "reasoning" : str(parsed.get("reasoning", "")).strip(),
        }
    except Exception as e:
        return None


def score_single_query(
    question_text : str,
    prompts       : dict,
    provider      : str,
    model         : str,
    client,
    question_idx  : int,
    total         : int
) -> dict:
    """
    Score one query on all three dimensions.
    Makes 3 LLM API calls — one per dimension.
    Retries on failure. Flags for manual review if all retries fail.

    Returns dict with scores and reasoning for all three dimensions.
    """
    results  = {}
    flagged  = []

    for dim_key in CONFIG["prompt_files"]:
        prompt = prompts[dim_key].replace('"{QUERY}"', f'"{question_text}"')
        # Also handle case where template uses {QUERY} without quotes
        prompt = prompt.replace("{QUERY}", question_text)

        parsed = None
        for attempt in range(CONFIG["max_retries"] + 1):
            try:
                raw    = call_llm(prompt, provider, model, client)
                parsed = parse_llm_response(raw, dim_key)
                if parsed is not None:
                    break
                else:
                    if attempt < CONFIG["max_retries"]:
                        time.sleep(CONFIG["retry_delay"])
            except Exception as e:
                if attempt < CONFIG["max_retries"]:
                    time.sleep(CONFIG["retry_delay"])
                else:
                    print(f"        ⚠ API error on {dim_key}: {e}")

        if parsed is None:
            # Flag for manual review — do not impute
            flagged.append(dim_key)
            results[dim_key] = {
                "dimension" : dim_key,
                "score"     : None,
                "reasoning" : "FLAGGED — LLM response could not be parsed after retries",
            }
        else:
            results[dim_key] = parsed

    return results, flagged


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SCORE ALL QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def score_all_queries(
    questions : list,
    prompts   : dict,
    provider  : str,
    model     : str,
    client,
    output_dir: str
) -> list:
    """
    Score all questions. Saves progress incrementally to handle interruptions.
    Returns list of per-query result dicts.
    """
    print(f"\n[2/5] Scoring {len(questions)} questions via LLM-as-judge")
    print(f"      Provider: {provider} | Model: {model}")
    print(f"      3 API calls per question = {len(questions)*3} total calls")
    print(f"      Progress saved incrementally to {output_dir}/per_query_scores.json\n")

    checkpoint_path = os.path.join(output_dir, "per_query_scores.json")

    # Load checkpoint if it exists (resume from interruption)
    completed_ids = set()
    all_results   = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        completed_ids = {r["question_id"] for r in all_results}
        print(f"      Resuming — {len(completed_ids)} questions already scored")

    total_flagged = []

    for i, q in enumerate(questions):
        qid  = q.get("question_id", str(i))
        text = q.get("question", "")
        qtype= q.get("question_type", "unknown")

        if qid in completed_ids:
            continue

        print(f"  [{i+1:3d}/{len(questions)}] {text[:70]}...")

        dim_results, flagged = score_single_query(
            question_text = text,
            prompts       = prompts,
            provider      = provider,
            model         = model,
            client        = client,
            question_idx  = i,
            total         = len(questions)
        )

        # Build per-query record
        scores = {
            k: v["score"] for k, v in dim_results.items()
        }
        valid_scores = [s for s in scores.values() if s is not None]
        composite    = round(sum(valid_scores) / len(valid_scores), 4) \
                       if valid_scores else None

        record = {
            "question_id"    : qid,
            "question"       : text,
            "question_type"  : qtype,
            "scores"         : scores,
            "reasoning"      : {k: v["reasoning"] for k, v in dim_results.items()},
            "composite_score": composite,
            "flagged_dims"   : flagged,
        }

        all_results.append(record)
        total_flagged.extend(flagged)

        score_str = " | ".join(
            f"{DIMENSION_LABELS[k]}={v['score'] or 'FLAG'}"
            for k, v in dim_results.items()
        )
        comp_str = f"{composite:.2f}" if composite else "N/A"
        print(f"         {score_str} | Composite={comp_str}")

        # Save checkpoint after every question
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f, indent=2)

    if total_flagged:
        print(f"\n      ⚠ {len(total_flagged)} dimension scores flagged for manual review")

    print(f"\n      Scoring complete. Results saved to {checkpoint_path}")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPUTE AGGREGATE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    """
    Compute aggregate metrics across all scored questions.
    Returns dict of metrics for the diagnostic report.
    """
    print(f"\n[3/5] Computing aggregate metrics...")

    n_total   = len(results)
    n_flagged = sum(1 for r in results if r["flagged_dims"])
    n_scored  = n_total - n_flagged

    # Per-dimension score distributions
    dim_distributions = {}
    for dim_key in CONFIG["prompt_files"]:
        scores = [
            r["scores"][dim_key]
            for r in results
            if r["scores"].get(dim_key) is not None
        ]
        counts = Counter(scores)
        dim_distributions[dim_key] = {
            "n_scored" : len(scores),
            "counts"   : {1: counts.get(1,0), 2: counts.get(2,0), 3: counts.get(3,0)},
            "mean"     : round(np.mean(scores), 3) if scores else None,
            "std"      : round(np.std(scores), 3) if scores else None,
            "pct_low"  : round(counts.get(1,0) / len(scores), 3) if scores else None,
            "pct_med"  : round(counts.get(2,0) / len(scores), 3) if scores else None,
            "pct_high" : round(counts.get(3,0) / len(scores), 3) if scores else None,
        }

    # Composite score distribution
    composites = [r["composite_score"] for r in results if r["composite_score"] is not None]

    def band(score):
        lo_e, hi_e = CONFIG["band_easy"]
        lo_m, hi_m = CONFIG["band_medium"]
        lo_h, hi_h = CONFIG["band_hard"]
        if lo_e <= score < hi_e:
            return "easy"
        elif lo_m <= score < hi_m:
            return "medium"
        elif lo_h <= score < hi_h:
            return "hard"
        return "uncategorized"

    band_counts = Counter(band(s) for s in composites)
    n_comp      = len(composites)

    composite_dist = {
        "n_scored"  : n_comp,
        "mean"      : round(np.mean(composites), 3) if composites else None,
        "std"       : round(np.std(composites), 3) if composites else None,
        "min"       : round(min(composites), 3) if composites else None,
        "max"       : round(max(composites), 3) if composites else None,
        "band_counts" : dict(band_counts),
        "band_pct"  : {
            b: round(band_counts.get(b,0) / n_comp, 3) if n_comp else 0
            for b in ["easy", "medium", "hard"]
        },
    }

    # Cross-tabulation: difficulty by question type
    type_breakdown = {}
    for qtype in set(r["question_type"] for r in results):
        type_scores = [
            r["composite_score"] for r in results
            if r["question_type"] == qtype
            and r["composite_score"] is not None
        ]
        type_breakdown[qtype] = {
            "n"    : len(type_scores),
            "mean" : round(np.mean(type_scores), 3) if type_scores else None,
            "std"  : round(np.std(type_scores), 3) if type_scores else None,
        }

    return {
        "n_total"           : n_total,
        "n_scored"          : n_scored,
        "n_flagged"         : n_flagged,
        "dim_distributions" : dim_distributions,
        "composite_dist"    : composite_dist,
        "type_breakdown"    : type_breakdown,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(results: list, metrics: dict, output_dir: str):
    """Generate all diagnostic visualizations."""
    print(f"\n[4/5] Generating visualizations...")

    composites = [r["composite_score"] for r in results if r["composite_score"] is not None]
    band_pct   = metrics["composite_dist"]["band_pct"]

    # ── Plot 1: Overall composite score distribution ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of composite scores
    ax = axes[0]
    bins = [1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0]
    ax.hist(composites, bins=bins, color="steelblue", edgecolor="white",
            linewidth=1.2, alpha=0.85)
    ax.axvline(x=1.67, color="orange", linestyle="--", linewidth=1.5,
               label="Easy/Medium boundary")
    ax.axvline(x=2.33, color="red", linestyle="--", linewidth=1.5,
               label="Medium/Hard boundary")
    ax.set_xlabel("Composite Difficulty Score", fontsize=12)
    ax.set_ylabel("Number of Questions", fontsize=12)
    ax.set_title("Overall Difficulty Distribution", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0.9, 3.1)

    # Band pie chart
    ax = axes[1]
    band_labels = ["Easy\n(1.0–1.67)", "Medium\n(1.67–2.33)", "Hard\n(2.33–3.0)"]
    band_sizes  = [
        metrics["composite_dist"]["band_counts"].get("easy",   0),
        metrics["composite_dist"]["band_counts"].get("medium", 0),
        metrics["composite_dist"]["band_counts"].get("hard",   0),
    ]
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    wedges, texts, autotexts = ax.pie(
        band_sizes, labels=band_labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 11}
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title("Difficulty Band Distribution", fontsize=13)

    plt.suptitle(
        f"Query Difficulty Diagnostic — {len(composites)} Questions\n"
        f"Mean composite: {metrics['composite_dist']['mean']:.2f} | "
        f"Std: {metrics['composite_dist']['std']:.2f}",
        fontsize=12, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "distribution_overall.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")

    # ── Plot 2: Per-dimension score distributions ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dim_colors = {"1": "#4CAF50", "2": "#FF9800", "3": "#F44336"}

    for ax, dim_key in zip(axes, CONFIG["prompt_files"]):
        dist    = metrics["dim_distributions"][dim_key]
        counts  = dist["counts"]
        labels  = ["Low (1)", "Medium (2)", "High (3)"]
        values  = [counts[1], counts[2], counts[3]]
        colors_ = ["#4CAF50", "#FF9800", "#F44336"]
        bars    = ax.bar(labels, values, color=colors_, edgecolor="white",
                         linewidth=1.2, alpha=0.85)

        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=10
                )

        ax.set_title(DIMENSION_LABELS[dim_key], fontsize=12)
        ax.set_ylabel("Number of Questions", fontsize=10)
        ax.set_xlabel("Score", fontsize=10)
        ax.set_ylim(0, max(values) * 1.2 + 2)

        mean_str = f"Mean: {dist['mean']:.2f}" if dist["mean"] else ""
        ax.text(0.97, 0.97, mean_str, transform=ax.transAxes,
                fontsize=9, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    plt.suptitle("Score Distribution by Dimension", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "distribution_by_dimension.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")

    # ── Plot 3: Difficulty by question type ───────────────────────────────────
    type_data = metrics["type_breakdown"]
    qtypes    = sorted(type_data.keys())
    means     = [type_data[t]["mean"] for t in qtypes]
    ns        = [type_data[t]["n"] for t in qtypes]
    stds      = [type_data[t]["std"] or 0 for t in qtypes]

    if len(qtypes) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        x      = np.arange(len(qtypes))
        bars   = ax.bar(x, means, yerr=stds, capsize=5,
                        color="steelblue", alpha=0.8, edgecolor="white")

        ax.axhline(y=metrics["composite_dist"]["mean"],
                   color="red", linestyle="--", linewidth=1.5,
                   label=f"Overall mean ({metrics['composite_dist']['mean']:.2f})")

        for bar, n in zip(bars, ns):
            ax.text(bar.get_x() + bar.get_width()/2,
                    0.05, f"n={n}", ha="center", va="bottom",
                    fontsize=9, color="white", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={n})" for t, n in zip(qtypes, ns)],
                           fontsize=11)
        ax.set_ylabel("Mean Composite Difficulty Score", fontsize=11)
        ax.set_title("Mean Difficulty by Question Type", fontsize=13)
        ax.set_ylim(1.0, 3.2)
        ax.legend(fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir, "distribution_by_type.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Saved: {path}")
    else:
        print(f"      Skipped type breakdown (only one question type found)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: GENERATE RECOMMENDATIONS AND REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(metrics: dict) -> list:
    """Threshold-triggered recommendations based on difficulty distribution."""
    recs = []
    bp   = metrics["composite_dist"]["band_pct"]
    bc   = metrics["composite_dist"]["band_counts"]
    n    = metrics["composite_dist"]["n_scored"]

    pct_easy   = bp.get("easy",   0)
    pct_medium = bp.get("medium", 0)
    pct_hard   = bp.get("hard",   0)
    n_easy     = bc.get("easy",   0)
    n_medium   = bc.get("medium", 0)
    n_hard     = bc.get("hard",   0)

    # Easy dominance
    if pct_easy > CONFIG["easy_maximum"]:
        recs.append(
            f"❌ SKEWED EASY: {pct_easy:.1%} of questions score in the easy band "
            f"({n_easy}/{n}). The question bank is dominated by low-difficulty queries. "
            f"This creates a ceiling effect — strong RAG systems cannot be "
            f"differentiated from weak ones on this bank. "
            f"Recommend adding at least {max(0, int(n*0.25) - n_hard)} hard questions."
        )
    elif pct_easy > 0.40:
        recs.append(
            f"⚠️  EASY-LEANING: {pct_easy:.1%} of questions are easy ({n_easy}/{n}). "
            f"Moderate skew toward low-difficulty queries. Consider adding more "
            f"medium and hard questions."
        )
    else:
        recs.append(
            f"✅ EASY BAND: {pct_easy:.1%} easy questions ({n_easy}/{n}). "
            f"Proportion is within acceptable range."
        )

    # Hard coverage
    if pct_hard < CONFIG["hard_minimum"]:
        recs.append(
            f"❌ INSUFFICIENT HARD QUESTIONS: Only {pct_hard:.1%} of questions "
            f"score in the hard band ({n_hard}/{n}). A valid measurement instrument "
            f"needs sufficient hard items to discriminate between high-performing "
            f"systems. Recommend a minimum of {max(15, int(n*0.20))} hard questions."
        )
    elif pct_hard < 0.25:
        recs.append(
            f"⚠️  FEW HARD QUESTIONS: {pct_hard:.1%} hard questions ({n_hard}/{n}). "
            f"Consider adding more complex, multi-hop, or exhaustive-answer queries."
        )
    else:
        recs.append(
            f"✅ HARD BAND: {pct_hard:.1%} hard questions ({n_hard}/{n}). "
            f"Good representation of difficult queries."
        )

    # Medium coverage
    if pct_medium < 0.20:
        recs.append(
            f"⚠️  THIN MIDDLE: Only {pct_medium:.1%} medium-difficulty questions "
            f"({n_medium}/{n}). The question bank lacks a gradient — it jumps from "
            f"easy to hard without sufficient intermediate difficulty."
        )
    else:
        recs.append(
            f"✅ MEDIUM BAND: {pct_medium:.1%} medium questions ({n_medium}/{n}). "
            f"Adequate intermediate difficulty coverage."
        )

    # Overall balance check
    ideal = 1.0 / 3
    max_deviation = max(
        abs(pct_easy - ideal),
        abs(pct_medium - ideal),
        abs(pct_hard - ideal)
    )
    if max_deviation < 0.10:
        recs.append(
            f"✅ WELL BALANCED: Difficulty distribution is close to ideal "
            f"(Easy {pct_easy:.1%} / Medium {pct_medium:.1%} / Hard {pct_hard:.1%}). "
            f"This question bank has good difficulty coverage."
        )
    elif max_deviation > 0.25:
        recs.append(
            f"❌ POOR BALANCE: Difficulty distribution deviates significantly "
            f"from ideal thirds "
            f"(Easy {pct_easy:.1%} / Medium {pct_medium:.1%} / Hard {pct_hard:.1%}). "
            f"The question bank does not provide balanced difficulty coverage."
        )

    # Flagged questions
    if metrics["n_flagged"] > 0:
        recs.append(
            f"ℹ️  MANUAL REVIEW NEEDED: {metrics['n_flagged']} questions could not "
            f"be scored by the LLM judge after retries. Check "
            f"per_query_scores.json for questions with flagged_dims set."
        )

    return recs


def save_outputs(
    results         : list,
    metrics         : dict,
    recommendations : list,
    provider        : str,
    model           : str,
    output_dir      : str
):
    """Save all outputs — JSON and human-readable report."""

    # ── difficulty_report.json ────────────────────────────────────────────────
    report = {
        "generated_at"    : datetime.utcnow().isoformat(),
        "llm_provider"    : provider,
        "llm_model"       : model,
        "n_total"         : metrics["n_total"],
        "n_scored"        : metrics["n_scored"],
        "n_flagged"       : metrics["n_flagged"],
        "composite_distribution" : metrics["composite_dist"],
        "dimension_distributions": metrics["dim_distributions"],
        "type_breakdown"  : metrics["type_breakdown"],
        "recommendations" : recommendations,
    }
    with open(os.path.join(output_dir, "difficulty_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── difficulty_report.txt ─────────────────────────────────────────────────
    cd  = metrics["composite_dist"]
    bp  = cd["band_pct"]
    bc  = cd["band_counts"]
    n   = cd["n_scored"]

    lines = [
        "=" * 65,
        "  QUERY DIFFICULTY DIAGNOSTIC REPORT",
        "  RAG Question Bank Audit — Stanford CS321M",
        f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  LLM Judge : {provider} / {model}",
        "=" * 65,
        "",
        "OVERALL DIFFICULTY DISTRIBUTION",
        "-" * 40,
        f"  Questions scored : {n}",
        f"  Questions flagged: {metrics['n_flagged']}",
        f"  Mean composite   : {cd['mean']:.2f}",
        f"  Std              : {cd['std']:.2f}",
        f"  Range            : {cd['min']:.2f} – {cd['max']:.2f}",
        "",
        "  DIFFICULTY BAND BREAKDOWN",
        f"  {'Band':<12} {'N':>6} {'%':>8}  {'Score Range'}",
        f"  {'-'*40}",
        f"  {'Easy':<12} {bc.get('easy',0):>6} {bp.get('easy',0):>7.1%}  1.00 – 1.67",
        f"  {'Medium':<12} {bc.get('medium',0):>6} {bp.get('medium',0):>7.1%}  1.67 – 2.33",
        f"  {'Hard':<12} {bc.get('hard',0):>6} {bp.get('hard',0):>7.1%}  2.33 – 3.00",
        "",
        "PER-DIMENSION SCORE DISTRIBUTIONS",
        "-" * 40,
    ]

    for dim_key, label in DIMENSION_LABELS.items():
        dist = metrics["dim_distributions"][dim_key]
        cnt  = dist["counts"]
        ns   = dist["n_scored"]
        lines += [
            f"  {label}",
            f"    Low  (1): {cnt[1]:3d} ({cnt[1]/ns:.1%})",
            f"    Med  (2): {cnt[2]:3d} ({cnt[2]/ns:.1%})",
            f"    High (3): {cnt[3]:3d} ({cnt[3]/ns:.1%})",
            f"    Mean: {dist['mean']:.2f} | Std: {dist['std']:.2f}",
            "",
        ]

    if len(metrics["type_breakdown"]) > 1:
        lines += ["DIFFICULTY BY QUESTION TYPE", "-" * 40]
        for qtype, tdata in sorted(metrics["type_breakdown"].items()):
            lines.append(
                f"  {qtype:<20} n={tdata['n']:3d} | "
                f"Mean={tdata['mean']:.2f} | Std={tdata['std']:.2f}"
            )
        lines.append("")

    lines += ["RECOMMENDATIONS", "-" * 40]
    for rec in recommendations:
        lines.append(f"  {rec}")
        lines.append("")

    lines += [
        "OUTPUT FILES",
        "-" * 40,
        "  per_query_scores.json        — per-question scores and reasoning",
        "  difficulty_report.json       — all aggregate metrics",
        "  difficulty_report.txt        — this report",
        "  distribution_overall.png     — composite score histogram + bands",
        "  distribution_by_dimension.png— per-dimension score bars",
        "  distribution_by_type.png     — difficulty by question type",
        "=" * 65,
    ]

    txt_path = os.path.join(output_dir, "difficulty_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n      All outputs saved to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    questions_path : str,
    prompts_dir    : str,
    provider       : str,
    model          : str,
    output_dir     : str
):
    print("\n" + "=" * 65)
    print("  QUERY DIFFICULTY DIAGNOSTIC")
    print("  Stanford CS321M — Phase 1 Question Bank Audit")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load inputs
    questions = load_questions(questions_path)
    prompts   = load_prompts(prompts_dir)

    # 2. Build LLM client
    print(f"\n[1.5/5] Initializing {provider} client...")
    if provider == "anthropic":
        client = build_anthropic_client()
    elif provider == "openai":
        client = build_openai_client()
    else:
        raise ValueError(f"Unknown provider: {provider}")
    print(f"        Client ready.")

    # 3. Score all queries
    results = score_all_queries(
        questions  = questions,
        prompts    = prompts,
        provider   = provider,
        model      = model,
        client     = client,
        output_dir = output_dir
    )

    # 4. Compute metrics
    metrics = compute_metrics(results)

    # 5. Visualizations
    generate_visualizations(results, metrics, output_dir)

    # 6. Recommendations + save
    recommendations = generate_recommendations(metrics)
    save_outputs(results, metrics, recommendations, provider, model, output_dir)

    # Print final summary
    cd = metrics["composite_dist"]
    bp = cd["band_pct"]
    bc = cd["band_counts"]

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Questions scored : {cd['n_scored']}")
    print(f"  Mean composite   : {cd['mean']:.2f} / 3.00")
    print(f"  Easy   : {bc.get('easy',0):3d} questions ({bp.get('easy',0):.1%})")
    print(f"  Medium : {bc.get('medium',0):3d} questions ({bp.get('medium',0):.1%})")
    print(f"  Hard   : {bc.get('hard',0):3d} questions ({bp.get('hard',0):.1%})")
    print()
    for rec in recommendations:
        print(f"  {rec}")
    print("=" * 65)

    return results, metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Query Difficulty Diagnostic for RAG Question Banks"
    )
    parser.add_argument(
        "--questions",
        default="data/processed/subsets/openrag_text_only_100.json",
        help="Path to question bank JSON file"
    )
    parser.add_argument(
        "--prompts_dir",
        default="src/analysis/prompts",
        help="Directory containing query_formulation.txt, reasoning_demand.txt, answer_form.txt"
    )
    parser.add_argument(
        "--llm_provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider for judging"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (defaults: claude-sonnet-4-20250514 for anthropic, gpt-4o for openai)"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/phase1/difficulty",
        help="Directory for all outputs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = args.model or CONFIG["default_models"][args.llm_provider]

    run_pipeline(
        questions_path = args.questions,
        prompts_dir    = args.prompts_dir,
        provider       = args.llm_provider,
        model          = model,
        output_dir     = args.output_dir,
    )
