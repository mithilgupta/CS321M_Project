"""
business_criticality_diagnostic.py
====================================
Automated Business Criticality Diagnostic for RAG Question Banks

Project : Toward Valid Internal Benchmarks for Enterprise RAG Evaluation
Course  : Stanford CS321M
Phase   : Phase 1 — Question Bank Audit

PURPOSE
-------
This script scores every question in a RAG question bank on its
business criticality — how harmful would an incorrect or hallucinated
answer be in an enterprise context.

DESIGN RATIONALE
----------------
Business criticality is explicitly NOT a property of the question's
linguistic complexity or corpus coverage. It is a property of the
DOMAIN the question belongs to and the CONSEQUENCES of a wrong answer.

This is the one dimension in our framework that requires domain
knowledge — in a real enterprise deployment, this map would be defined
by Subject Matter Experts (SMEs) who understand which topics carry
legal, financial, safety, or operational risk.

For this study, we demonstrate the mechanism with an illustrative
domain-criticality map. The LLM-as-judge classifies each query into
a domain and assigns the corresponding criticality score. This shows
HOW SME input would be incorporated — the map is the only thing that
changes per enterprise.

THE THREE CRITICALITY LEVELS
-----------------------------
  3 — HIGH: Wrong answer carries legal, financial, safety, or
      compliance risk. Incorrect information could cause direct harm
      to the organization or its stakeholders.

  2 — MEDIUM: Wrong answer causes operational disruption or customer
      dissatisfaction but does not create legal or safety exposure.

  1 — LOW: Wrong answer causes minor inconvenience. Informational
      queries where errors are easily caught and corrected.

ILLUSTRATIVE DOMAIN MAP (replace with SME input in production)
--------------------------------------------------------------
  HIGH (3): Compliance, Regulatory, Legal, Pricing, Financial,
            Safety, Medical, Security, Data Privacy
  MEDIUM (2): Product specs, Technical documentation, HR policy,
              Customer contracts, SLA commitments
  LOW (1): General FAQ, Onboarding, Internal knowledge base,
           Informational queries

INPUT
-----
  --questions     : path to question bank JSON
  --prompts_dir   : directory containing criticality_scoring.txt
  --llm_provider  : anthropic | openai
  --model         : model name
  --output_dir    : directory for all outputs

OUTPUT
------
  per_query_scores.json       per-question domain + criticality score
  criticality_report.json     aggregate metrics
  criticality_report.txt      human-readable diagnostic report
  distribution_criticality.png  criticality level distribution
  domain_breakdown.png          queries per domain
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
    # Criticality levels
    "score_min" : 1,
    "score_max" : 3,

    # Domain → criticality mapping (illustrative — replace with SME input)
    "domain_criticality_map" : {
        # HIGH criticality domains
        "Compliance and Regulatory"  : 3,
        "Legal and Contractual"      : 3,
        "Pricing and Financial"      : 3,
        "Safety and Risk"            : 3,
        "Data Privacy and Security"  : 3,
        "Medical and Health"         : 3,

        # MEDIUM criticality domains
        "Product and Technical Specs": 2,
        "HR and People Policy"       : 2,
        "Customer Contracts and SLA" : 2,
        "Engineering and Operations" : 2,
        "Research and Development"   : 2,

        # LOW criticality domains
        "General FAQ"                : 1,
        "Onboarding and Training"    : 1,
        "Internal Knowledge Base"    : 1,
        "Informational"              : 1,
    },

    # Thresholds for recommendations
    "high_minimum"   : 0.15,   # <15% high criticality = may be missing risk coverage
    "low_maximum"    : 0.70,   # >70% low criticality = bank is trivial

    # LLM settings
    "max_retries"  : 2,
    "retry_delay"  : 2.0,
    "temperature"  : 0.0,
    "max_tokens"   : 600,

    # Prompt file
    "prompt_file" : "criticality_scoring.txt",

    # Default models
    "default_models" : {
        "anthropic" : "claude-sonnet-4-5",
        "openai"    : "gpt-4o",
    },
}

CRITICALITY_LABELS = {1: "Low", 2: "Medium", 3: "High"}
CRITICALITY_COLORS = {1: "#4CAF50", 2: "#FF9800", 3: "#F44336"}


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT TEMPLATE (written inline — no external file needed)
# ─────────────────────────────────────────────────────────────────────────────

CRITICALITY_PROMPT_TEMPLATE = '''You are an expert in enterprise risk assessment and RAG system evaluation.

Your task is to classify a query from an enterprise RAG system into the domain it belongs to,
and assign a business criticality score based on how harmful an incorrect answer would be.

---

DOMAIN LIST AND CRITICALITY SCORES:

HIGH CRITICALITY (Score 3) — Wrong answer carries legal, financial, safety, or compliance risk:
  - Compliance and Regulatory: questions about regulations, audits, legal requirements
  - Legal and Contractual: questions about contracts, liability, legal obligations
  - Pricing and Financial: questions about pricing, costs, financial figures, budgets
  - Safety and Risk: questions about safety protocols, hazard procedures, risk management
  - Data Privacy and Security: questions about data handling, GDPR, access controls, security
  - Medical and Health: questions about medical procedures, health guidelines, clinical decisions

MEDIUM CRITICALITY (Score 2) — Wrong answer causes operational disruption or customer impact:
  - Product and Technical Specs: questions about product features, technical specifications
  - HR and People Policy: questions about HR policies, benefits, performance management
  - Customer Contracts and SLA: questions about customer commitments, service levels
  - Engineering and Operations: questions about system architecture, operational procedures
  - Research and Development: questions about research findings, experimental results

LOW CRITICALITY (Score 1) — Wrong answer causes minor inconvenience, easily corrected:
  - General FAQ: common informational questions with widely known answers
  - Onboarding and Training: questions about getting started, learning resources
  - Internal Knowledge Base: general internal reference questions
  - Informational: background information, definitions, general explanations

---

INSTRUCTIONS:
1. Read the query carefully.
2. Identify which domain best describes this query.
3. Assign the criticality score corresponding to that domain.
4. Explain in 2-3 sentences why this domain classification is appropriate and what the
   consequences of a wrong answer would be in an enterprise context.
5. Respond in this exact JSON format — no other text:

{{
  "domain": "exact domain name from the list above",
  "criticality_score": 1,
  "reasoning": "2-3 sentences explaining the domain classification and why a wrong answer would have this level of business impact"
}}

---

QUERY TO CLASSIFY:
"{QUERY}"
'''


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_questions(questions_path: str) -> list:
    print(f"\n[1/4] Loading questions from {questions_path}")
    with open(questions_path) as f:
        questions = json.load(f)
    print(f"      Questions loaded: {len(questions)}")
    assert len(questions) > 0, "Question file is empty"
    return questions


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: LLM JUDGE
# ─────────────────────────────────────────────────────────────────────────────

def build_client(provider: str):
    if provider == "anthropic":
        try:
            import anthropic
            return anthropic.Anthropic()
        except ImportError:
            raise ImportError("Run: pip install anthropic")
    elif provider == "openai":
        try:
            import openai
            return openai.OpenAI()
        except ImportError:
            raise ImportError("Run: pip install openai")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def call_llm(prompt: str, provider: str, model: str, client) -> str:
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


def parse_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines   = cleaned.split("\n")
        cleaned = "\n".join(l for l in lines if not l.strip().startswith("```")).strip()
    if not cleaned.startswith("{"):
        start   = cleaned.find("{")
        end     = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            cleaned = cleaned[start:end]
    try:
        parsed = json.loads(cleaned)
        score  = int(parsed.get("criticality_score", -1))
        domain = str(parsed.get("domain", "Unknown"))
        assert score in [1, 2, 3], f"Score {score} not in [1,2,3]"
        return {
            "domain"            : domain,
            "criticality_score" : score,
            "reasoning"         : str(parsed.get("reasoning", "")).strip(),
        }
    except Exception:
        return None


def score_single_query(
    question_text : str,
    provider      : str,
    model         : str,
    client
) -> dict:
    prompt = CRITICALITY_PROMPT_TEMPLATE.replace('"{QUERY}"', f'"{question_text}"')
    prompt = prompt.replace("{QUERY}", question_text)

    parsed = None
    for attempt in range(CONFIG["max_retries"] + 1):
        try:
            raw    = call_llm(prompt, provider, model, client)
            parsed = parse_response(raw)
            if parsed is not None:
                break
            if attempt < CONFIG["max_retries"]:
                time.sleep(CONFIG["retry_delay"])
        except Exception as e:
            if attempt < CONFIG["max_retries"]:
                time.sleep(CONFIG["retry_delay"])
            else:
                print(f"        ⚠ API error: {e}")

    if parsed is None:
        return {
            "domain"            : "FLAGGED",
            "criticality_score" : None,
            "reasoning"         : "FLAGGED — could not parse LLM response after retries",
            "flagged"           : True,
        }

    parsed["flagged"] = False
    return parsed


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: SCORE ALL QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def score_all_queries(
    questions  : list,
    provider   : str,
    model      : str,
    client,
    output_dir : str
) -> list:
    print(f"\n[2/4] Scoring {len(questions)} questions via LLM-as-judge")
    print(f"      Provider: {provider} | Model: {model}")
    print(f"      1 API call per question = {len(questions)} total calls\n")

    checkpoint_path = os.path.join(output_dir, "per_query_scores.json")

    # Resume from checkpoint if available
    completed_ids = set()
    all_results   = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        completed_ids = {r["question_id"] for r in all_results}
        print(f"      Resuming — {len(completed_ids)} questions already scored")

    for i, q in enumerate(questions):
        qid   = q.get("question_id", str(i))
        text  = q.get("question", "")
        qtype = q.get("question_type", "unknown")

        if qid in completed_ids:
            continue

        print(f"  [{i+1:3d}/{len(questions)}] {text[:70]}...")

        result = score_single_query(text, provider, model, client)

        record = {
            "question_id"       : qid,
            "question"          : text,
            "question_type"     : qtype,
            "domain"            : result["domain"],
            "criticality_score" : result["criticality_score"],
            "reasoning"         : result["reasoning"],
            "flagged"           : result.get("flagged", False),
        }

        all_results.append(record)

        score_str = result["criticality_score"] or "FLAG"
        label_str = CRITICALITY_LABELS.get(result["criticality_score"], "FLAGGED")
        print(f"         Domain: {result['domain']} | Score: {score_str} ({label_str})")

        # Save checkpoint after every question
        with open(checkpoint_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n      Scoring complete.")
    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: COMPUTE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: list) -> dict:
    print(f"\n[3/4] Computing aggregate metrics...")

    n_total   = len(results)
    n_flagged = sum(1 for r in results if r.get("flagged"))
    scores    = [r["criticality_score"] for r in results if r["criticality_score"] is not None]
    n_scored  = len(scores)

    counts = Counter(scores)
    domain_counts = Counter(r["domain"] for r in results if not r.get("flagged"))

    score_dist = {
        "n_scored"  : n_scored,
        "counts"    : {1: counts.get(1,0), 2: counts.get(2,0), 3: counts.get(3,0)},
        "pct_low"   : round(counts.get(1,0) / n_scored, 3) if n_scored else 0,
        "pct_medium": round(counts.get(2,0) / n_scored, 3) if n_scored else 0,
        "pct_high"  : round(counts.get(3,0) / n_scored, 3) if n_scored else 0,
        "mean"      : round(np.mean(scores), 3) if scores else None,
        "std"       : round(np.std(scores), 3) if scores else None,
    }

    return {
        "n_total"      : n_total,
        "n_scored"     : n_scored,
        "n_flagged"    : n_flagged,
        "score_dist"   : score_dist,
        "domain_counts": dict(domain_counts),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(results: list, metrics: dict, output_dir: str):
    print(f"\n[4a/4] Generating visualizations...")

    sd = metrics["score_dist"]

    # ── Plot 1: Criticality distribution ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    labels = ["Low (1)", "Medium (2)", "High (3)"]
    values = [sd["counts"][1], sd["counts"][2], sd["counts"][3]]
    colors = [CRITICALITY_COLORS[1], CRITICALITY_COLORS[2], CRITICALITY_COLORS[3]]
    bars   = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, alpha=0.85)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Business Criticality Distribution", fontsize=13)
    ax.set_ylabel("Number of Questions", fontsize=11)
    ax.set_xlabel("Criticality Level", fontsize=11)
    ax.set_ylim(0, max(values) * 1.2 + 2)

    # Pie chart
    ax = axes[1]
    pie_sizes  = [sd["counts"][1], sd["counts"][2], sd["counts"][3]]
    pie_labels = [f"Low\n({sd['pct_low']:.1%})",
                  f"Medium\n({sd['pct_medium']:.1%})",
                  f"High\n({sd['pct_high']:.1%})"]
    pie_colors = [CRITICALITY_COLORS[1], CRITICALITY_COLORS[2], CRITICALITY_COLORS[3]]
    wedges, texts, autotexts = ax.pie(
        pie_sizes, labels=pie_labels, colors=pie_colors,
        autopct="%1.0f%%", startangle=90
    )
    ax.set_title("Criticality Band Breakdown", fontsize=13)

    plt.suptitle(
        f"Business Criticality Diagnostic — {sd['n_scored']} Questions\n"
        f"Mean: {sd['mean']:.2f} | High: {sd['pct_high']:.1%} | "
        f"Medium: {sd['pct_medium']:.1%} | Low: {sd['pct_low']:.1%}",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "distribution_criticality.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {path}")

    # ── Plot 2: Domain breakdown ──────────────────────────────────────────────
    domain_counts = metrics["domain_counts"]
    if domain_counts:
        sorted_domains = sorted(domain_counts.items(), key=lambda x: -x[1])
        domains = [d[0] for d in sorted_domains]
        counts  = [d[1] for d in sorted_domains]

        # Color bars by criticality level of each domain
        bar_colors = []
        for d in domains:
            score = CONFIG["domain_criticality_map"].get(d, 1)
            bar_colors.append(CRITICALITY_COLORS[score])

        fig, ax = plt.subplots(figsize=(12, max(5, len(domains) * 0.5)))
        bars = ax.barh(domains, counts, color=bar_colors, edgecolor="white",
                       linewidth=1.0, alpha=0.85)
        for bar, val in zip(bars, counts):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    str(val), va="center", fontsize=9)

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=CRITICALITY_COLORS[3], label="High Criticality"),
            Patch(facecolor=CRITICALITY_COLORS[2], label="Medium Criticality"),
            Patch(facecolor=CRITICALITY_COLORS[1], label="Low Criticality"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
        ax.set_xlabel("Number of Questions", fontsize=11)
        ax.set_title("Questions by Domain and Criticality", fontsize=13)
        ax.invert_yaxis()
        plt.tight_layout()
        path = os.path.join(output_dir, "domain_breakdown.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"      Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(metrics: dict) -> list:
    recs   = []
    sd     = metrics["score_dist"]
    n      = sd["n_scored"]

    pct_high   = sd["pct_high"]
    pct_medium = sd["pct_medium"]
    pct_low    = sd["pct_low"]
    n_high     = sd["counts"][3]
    n_medium   = sd["counts"][2]
    n_low      = sd["counts"][1]

    # High criticality coverage
    if pct_high < CONFIG["high_minimum"]:
        recs.append(
            f"❌ INSUFFICIENT HIGH-CRITICALITY COVERAGE: Only {pct_high:.1%} of questions "
            f"({n_high}/{n}) cover high-criticality domains. The question bank may be "
            f"under-testing the areas where RAG failures carry the most business risk. "
            f"Add questions covering compliance, legal, pricing, and safety domains."
        )
    else:
        recs.append(
            f"✅ HIGH-CRITICALITY COVERAGE: {pct_high:.1%} of questions ({n_high}/{n}) "
            f"cover high-criticality domains. Adequate risk coverage."
        )

    # Low criticality dominance
    if pct_low > CONFIG["low_maximum"]:
        recs.append(
            f"⚠️  LOW-CRITICALITY DOMINATED: {pct_low:.1%} of questions ({n_low}/{n}) "
            f"are low-criticality informational queries. The question bank may not "
            f"adequately stress-test the RAG system on business-critical topics."
        )
    else:
        recs.append(
            f"✅ CRITICALITY BALANCE: Low-criticality questions at {pct_low:.1%} — "
            f"within acceptable range."
        )

    # Note on SME input
    recs.append(
        f"ℹ️  SME INPUT REQUIRED: The domain-criticality map used in this analysis is "
        f"illustrative. In production, this map should be defined by Subject Matter "
        f"Experts who understand which query domains carry the highest business risk "
        f"for this specific enterprise context."
    )

    if metrics["n_flagged"] > 0:
        recs.append(
            f"ℹ️  MANUAL REVIEW: {metrics['n_flagged']} questions could not be scored. "
            f"Check per_query_scores.json for flagged entries."
        )

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    results         : list,
    metrics         : dict,
    recommendations : list,
    provider        : str,
    model           : str,
    output_dir      : str
):
    # criticality_report.json
    report = {
        "generated_at"    : datetime.utcnow().isoformat(),
        "llm_provider"    : provider,
        "llm_model"       : model,
        "domain_map_note" : "Illustrative domain-criticality map. Replace with SME input in production.",
        "n_total"         : metrics["n_total"],
        "n_scored"        : metrics["n_scored"],
        "n_flagged"       : metrics["n_flagged"],
        "score_distribution": metrics["score_dist"],
        "domain_counts"   : metrics["domain_counts"],
        "recommendations" : recommendations,
    }
    with open(os.path.join(output_dir, "criticality_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # criticality_report.txt
    sd = metrics["score_dist"]
    lines = [
        "=" * 65,
        "  BUSINESS CRITICALITY DIAGNOSTIC REPORT",
        "  RAG Question Bank Audit — Stanford CS321M",
        f"  Generated : {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        f"  LLM Judge : {provider} / {model}",
        "  NOTE: Domain map is illustrative. SME input required for production.",
        "=" * 65,
        "",
        "CRITICALITY DISTRIBUTION",
        "-" * 40,
        f"  Questions scored : {sd['n_scored']}",
        f"  Mean criticality : {sd['mean']:.2f}",
        f"",
        f"  {'Level':<12} {'N':>6} {'%':>8}",
        f"  {'-'*30}",
        f"  {'High (3)':<12} {sd['counts'][3]:>6} {sd['pct_high']:>7.1%}",
        f"  {'Medium (2)':<12} {sd['counts'][2]:>6} {sd['pct_medium']:>7.1%}",
        f"  {'Low (1)':<12} {sd['counts'][1]:>6} {sd['pct_low']:>7.1%}",
        "",
        "DOMAIN BREAKDOWN",
        "-" * 40,
    ]
    for domain, count in sorted(metrics["domain_counts"].items(), key=lambda x: -x[1]):
        score = CONFIG["domain_criticality_map"].get(domain, "?")
        lines.append(f"  {domain:<35} {count:>4} questions  (criticality={score})")

    lines += ["", "RECOMMENDATIONS", "-" * 40]
    for rec in recommendations:
        lines.append(f"  {rec}")
        lines.append("")

    lines += [
        "OUTPUT FILES",
        "-" * 40,
        "  per_query_scores.json      — per-question domain + criticality",
        "  criticality_report.json    — aggregate metrics",
        "  criticality_report.txt     — this report",
        "  distribution_criticality.png — criticality level distribution",
        "  domain_breakdown.png       — questions by domain",
        "=" * 65,
    ]

    with open(os.path.join(output_dir, "criticality_report.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n      All outputs saved to: {output_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    questions_path : str,
    provider       : str,
    model          : str,
    output_dir     : str
):
    print("\n" + "=" * 65)
    print("  BUSINESS CRITICALITY DIAGNOSTIC")
    print("  Stanford CS321M — Phase 1 Question Bank Audit")
    print("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    questions = load_questions(questions_path)

    print(f"\n[1.5/4] Initializing {provider} client...")
    client = build_client(provider)
    print(f"        Client ready.")

    results = score_all_queries(questions, provider, model, client, output_dir)
    metrics = compute_metrics(results)
    generate_visualizations(results, metrics, output_dir)
    recommendations = generate_recommendations(metrics)
    save_outputs(results, metrics, recommendations, provider, model, output_dir)

    # Print summary
    sd = metrics["score_dist"]
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Questions scored : {sd['n_scored']}")
    print(f"  Mean criticality : {sd['mean']:.2f} / 3.00")
    print(f"  High   (3): {sd['counts'][3]:3d} questions ({sd['pct_high']:.1%})")
    print(f"  Medium (2): {sd['counts'][2]:3d} questions ({sd['pct_medium']:.1%})")
    print(f"  Low    (1): {sd['counts'][1]:3d} questions ({sd['pct_low']:.1%})")
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
        description="Business Criticality Diagnostic for RAG Question Banks"
    )
    parser.add_argument(
        "--questions",
        default="data/processed/subsets/openrag_text_only_100.json",
        help="Path to question bank JSON file"
    )
    parser.add_argument(
        "--llm_provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="LLM provider"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/phase1/criticality",
        help="Directory for all outputs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args  = parse_args()
    model = args.model or CONFIG["default_models"][args.llm_provider]
    run_pipeline(
        questions_path = args.questions,
        provider       = args.llm_provider,
        model          = model,
        output_dir     = args.output_dir,
    )
