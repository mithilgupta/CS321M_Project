"""
semantic_coverage_diagnostic.py
================================
Automated Semantic Coverage Diagnostic for RAG Question Banks

Project : Toward Valid Internal Benchmarks for Enterprise RAG Evaluation
Course  : Stanford CS321M
Phase   : Phase 1 — Question Bank Audit

PURPOSE
-------
This script answers two research questions about a RAG question bank:

  RQ1 — CORPUS COVERAGE (Content Validity)
        Does the question bank cover the document corpus, or are there
        large semantic regions that no question ever touches?

  RQ2 — QUERY DIVERSITY (Construct Validity)
        Are the questions semantically diverse, or are they clustered
        into a small number of redundant groups?

METHOD
------
  1. Load pre-computed corpus embeddings from FAISS index
  2. Embed queries using the same model (shared vector space)
  3. UMAP 10D dimensionality reduction on corpus (preserves local structure)
  4. HDBSCAN clustering on UMAP-reduced corpus
     - No manual k selection required (improves on Bröstl et al. 2025,
       Klearman et al. 2026 which both use k-means)
     - Handles arbitrary cluster shapes and explicit noise
  5. Assign queries to clusters via HDBSCAN soft membership
  6. Compute Coverage Score (0-10)
  7. Compute Diversity Score (0-10)
  8. Generate visualizations and recommendations

INPUT
-----
  --corpus_embeddings  : path to .npy file of corpus chunk embeddings
  --queries            : path to .json file of question bank
  --metadata           : path to .jsonl file of corpus chunk metadata
  --embedder_name      : path to .txt file containing embedder model name
  --output_dir         : directory for all outputs (default: outputs/phase1/coverage)

OUTPUT
------
  coverage_score           (0-10)
  diversity_score          (0-10)
  diagnostic_report.json   all metrics
  blind_spot_clusters.json representative chunks per uncovered cluster
  near_duplicate_pairs.json query pairs with cosine similarity >= threshold
  umap_2d_coverage.png     corpus + query overlay visualization
  hdbscan_coverage_map.png cluster map with coverage annotations
  query_spread_map.png     query cluster visualization (RQ2)
  diagnostic_report.txt    human-readable report with recommendations

REFERENCES
----------
  - Bröstl et al. (2025) arXiv:2510.00001 — prior work using k-means
  - Klearman et al. (2026) arXiv:2604.20763 — prior work using k-means
  - McInnes et al. (2018) UMAP — dimensionality reduction
  - McInnes et al. (2017) HDBSCAN — density-based clustering
  - Grootendorst (2022) BERTopic — UMAP + HDBSCAN pipeline for text
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ── Optional heavy imports — checked at runtime ──────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — all tunable parameters in one place
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # UMAP parameters
    "umap_n_components_2d"  : 2,      # for visualization
    "umap_n_components_hd"  : 10,     # for HDBSCAN input
    "umap_n_neighbors"      : 15,     # local neighborhood size
    "umap_min_dist_2d"      : 0.1,    # looser for visualization
    "umap_min_dist_hd"      : 0.0,    # tighter for clustering
    "umap_metric"           : "cosine",
    "umap_random_state"     : 42,

    # HDBSCAN parameters
    "hdbscan_min_cluster_size"       : 50,
    "hdbscan_min_samples"            : 5,
    "hdbscan_metric"                 : "euclidean",  # in UMAP-reduced space
    "hdbscan_cluster_selection"      : "eom",
    "hdbscan_noise_membership_threshold" : 0.1,

    # Diversity parameters
    "near_duplicate_threshold" : 0.85,   # cosine similarity threshold

    # Score thresholds (for recommendations)
    "coverage_thresholds"  : {"good": 0.8, "warning": 0.5},
    "diversity_thresholds" : {"good": 7.0, "warning": 4.0},

    # Random seed
    "random_state" : 42,
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD INPUTS
# ─────────────────────────────────────────────────────────────────────────────

def load_corpus_embeddings(embeddings_path: str) -> np.ndarray:
    """
    Load pre-computed corpus embeddings from .npy file.
    These were extracted from the FAISS index — we do NOT re-embed.
    """
    print(f"\n[1/7] Loading corpus embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"      Corpus embeddings shape: {embeddings.shape}")
    assert embeddings.ndim == 2, "Expected 2D array (n_chunks × dim)"
    return embeddings.astype("float32")


def load_corpus_metadata(metadata_path: str) -> list:
    """Load corpus chunk metadata (title, text, doc_id per chunk)."""
    print(f"      Loading corpus metadata from {metadata_path}")
    metadata = []
    with open(metadata_path) as f:
        for line in f:
            metadata.append(json.loads(line.strip()))
    print(f"      Metadata records: {len(metadata)}")
    assert len(metadata) > 0, "Metadata file is empty"
    return metadata


def load_questions(questions_path: str) -> list:
    """Load question bank from JSON file."""
    print(f"      Loading questions from {questions_path}")
    with open(questions_path) as f:
        questions = json.load(f)
    print(f"      Questions loaded: {len(questions)}")
    assert len(questions) > 0, "Question file is empty"
    return questions


def embed_queries(
    questions: list,
    embedder_name_path: str
) -> tuple:
    """
    Embed all queries using the SAME model used for the corpus.
    This is critical — queries and corpus must share a vector space.

    Returns:
        query_embeddings : np.ndarray (n_queries × dim)
        query_texts      : list of question strings
        query_ids        : list of question IDs
    """
    print(f"\n[2/7] Embedding queries")
    with open(embedder_name_path) as f:
        model_name = f.read().strip()
    print(f"      Using embedder: {model_name}")

    embedder = SentenceTransformer(model_name)
    query_texts = [q["question"] for q in questions]
    query_ids   = [q.get("question_id", str(i)) for i, q in enumerate(questions)]

    query_embeddings = embedder.encode(
        query_texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    ).astype("float32")

    print(f"      Query embeddings shape: {query_embeddings.shape}")
    assert query_embeddings.shape[1] == query_embeddings.shape[1], \
        "Query and corpus embedding dimensions must match"
    return query_embeddings, query_texts, query_ids


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: DIMENSIONALITY REDUCTION (UMAP)
# ─────────────────────────────────────────────────────────────────────────────

def run_umap_2d(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray
) -> tuple:
    """
    UMAP 2D reduction for visualization only.
    Fit on corpus, transform queries into the same 2D space.

    Why UMAP over PCA:
      PCA is linear and destroys local cluster structure in high-dimensional
      embedding spaces. UMAP preserves local neighborhood structure, making
      it far more informative for visualizing semantic clusters.
      (McInnes et al., 2018; Grootendorst, 2022 BERTopic)

    Returns:
        corpus_2d : np.ndarray (n_chunks × 2)
        query_2d  : np.ndarray (n_queries × 2)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    print(f"\n[3/7] UMAP 2D reduction (visualization) — this takes 2-4 min...")
    reducer_2d = umap.UMAP(
        n_components  = CONFIG["umap_n_components_2d"],
        n_neighbors   = CONFIG["umap_n_neighbors"],
        min_dist      = CONFIG["umap_min_dist_2d"],
        metric        = CONFIG["umap_metric"],
        random_state  = CONFIG["umap_random_state"],
        verbose       = False
    )
    corpus_2d = reducer_2d.fit_transform(corpus_embeddings)
    query_2d  = reducer_2d.transform(query_embeddings)
    print(f"      2D reduction complete.")
    return corpus_2d, query_2d


def run_umap_hd(
    corpus_embeddings: np.ndarray,
    query_embeddings: np.ndarray
) -> tuple:
    """
    UMAP high-dimensional reduction for HDBSCAN input.
    10D preserves more structure than 2D, giving HDBSCAN better signal.
    min_dist=0.0 encourages tighter clusters — better HDBSCAN input.

    Returns:
        corpus_hd : np.ndarray (n_chunks × 10)
        query_hd  : np.ndarray (n_queries × 10)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    print(f"      UMAP {CONFIG['umap_n_components_hd']}D reduction "
          f"(HDBSCAN input) — this takes 2-3 min...")
    reducer_hd = umap.UMAP(
        n_components  = CONFIG["umap_n_components_hd"],
        n_neighbors   = CONFIG["umap_n_neighbors"],
        min_dist      = CONFIG["umap_min_dist_hd"],
        metric        = CONFIG["umap_metric"],
        random_state  = CONFIG["umap_random_state"],
        verbose       = False
    )
    corpus_hd = reducer_hd.fit_transform(corpus_embeddings)
    query_hd  = reducer_hd.transform(query_embeddings)
    print(f"      {CONFIG['umap_n_components_hd']}D reduction complete.")
    return corpus_hd, query_hd


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: HDBSCAN CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────

def run_hdbscan(corpus_hd: np.ndarray) -> object:
    """
    Cluster the corpus using HDBSCAN on UMAP-reduced embeddings.

    Why HDBSCAN over k-means:
      1. No k required — cluster count emerges from data density
      2. Handles arbitrary cluster shapes (semantic topics are not spherical)
      3. Explicit noise handling — low-density regions marked as noise
      4. Our elbow analysis (k=10 to k=500) showed nearly linear inertia,
         confirming the corpus lacks the discrete spherical cluster structure
         k-means assumes.
      (McInnes et al., 2017; Asyaky & Mandala, 2021)

    Returns:
        clusterer : fitted HDBSCAN object
    """
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan not installed. Run: pip install hdbscan")

    print(f"\n[4/7] HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size       = CONFIG["hdbscan_min_cluster_size"],
        min_samples            = CONFIG["hdbscan_min_samples"],
        metric                 = CONFIG["hdbscan_metric"],
        cluster_selection_method = CONFIG["hdbscan_cluster_selection"],
        prediction_data        = True  # Required for soft membership on queries
    )
    clusterer.fit(corpus_hd)

    corpus_labels = clusterer.labels_
    n_clusters = len(set(corpus_labels)) - (1 if -1 in corpus_labels else 0)
    n_noise    = (corpus_labels == -1).sum()

    print(f"      Clusters found:  {n_clusters}")
    print(f"      Noise points:    {n_noise} ({n_noise/len(corpus_labels):.1%})")

    cluster_sizes = Counter(corpus_labels)
    for label in sorted(cluster_sizes):
        tag   = "NOISE" if label == -1 else f"Cluster {label:3d}"
        count = cluster_sizes[label]
        print(f"        {tag}: {count:5d} chunks")

    return clusterer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ASSIGN QUERIES TO CLUSTERS
# ─────────────────────────────────────────────────────────────────────────────

def assign_queries_to_clusters(
    clusterer,
    query_hd: np.ndarray
) -> tuple:
    """
    Assign each query to the most probable HDBSCAN cluster using
    soft membership vectors.

    Soft membership gives each query a probability vector over all clusters.
    Assignment = argmax (most probable cluster).
    Queries with max probability < threshold are flagged as "noise territory"
    — they don't strongly belong to any identified cluster.

    Returns:
        query_assignments : np.ndarray (n_queries,) — cluster ID per query
        query_is_noise    : np.ndarray (n_queries,) — bool, True = noise territory
        soft_clusters     : np.ndarray (n_queries × n_clusters) — membership matrix
    """
    print(f"\n[5/7] Assigning queries to clusters...")

    soft_clusters     = hdbscan.membership_vector(clusterer, query_hd)
    query_assignments = np.argmax(soft_clusters, axis=1)
    query_max_membership = soft_clusters.max(axis=1)
    query_is_noise    = query_max_membership < CONFIG["hdbscan_noise_membership_threshold"]

    print(f"      Queries assigned to clusters: {(~query_is_noise).sum()}")
    print(f"      Queries in noise territory:   {query_is_noise.sum()}")
    return query_assignments, query_is_noise, soft_clusters


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: COVERAGE SCORE (RQ1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_coverage_score(
    clusterer,
    query_assignments: np.ndarray,
    query_is_noise: np.ndarray,
    corpus_embeddings: np.ndarray,
    metadata: list
) -> dict:
    """
    RQ1: What fraction of corpus semantic clusters has at least one query?

    Coverage Score = (clusters with ≥1 query) / total_clusters × 10

    Blind spots = clusters with zero queries. We characterize each blind
    spot by finding the 3 chunks closest to the cluster centroid, giving
    a human-readable description of what topic is being missed.

    Returns dict with all RQ1 metrics.
    """
    print(f"\n[6a/7] Computing Coverage Score (RQ1)...")

    corpus_labels = clusterer.labels_
    n_clusters    = len(set(corpus_labels)) - (1 if -1 in corpus_labels else 0)

    # Which clusters have at least one assigned query?
    assigned_query_clusters = set(query_assignments[~query_is_noise].tolist())
    all_cluster_ids         = set(range(n_clusters))
    covered_clusters        = assigned_query_clusters & all_cluster_ids
    blind_spot_clusters     = all_cluster_ids - covered_clusters

    coverage_rate  = len(covered_clusters) / n_clusters if n_clusters > 0 else 0.0
    coverage_score = coverage_rate * 10.0

    print(f"      Total clusters:           {n_clusters}")
    print(f"      Clusters covered:         {len(covered_clusters)}")
    print(f"      Blind spot clusters:      {len(blind_spot_clusters)}")
    print(f"      Coverage rate:            {coverage_rate:.1%}")
    print(f"      Coverage score (0-10):    {coverage_score:.2f}")

    # Characterize blind spot clusters
    blind_spot_summaries = []
    for cluster_id in sorted(blind_spot_clusters):
        chunk_indices = np.where(corpus_labels == cluster_id)[0]
        if len(chunk_indices) == 0:
            continue

        # Compute centroid of this cluster
        cluster_vecs = corpus_embeddings[chunk_indices]
        centroid     = cluster_vecs.mean(axis=0)

        # Find 3 chunks closest to centroid (most representative)
        distances = np.linalg.norm(cluster_vecs - centroid, axis=1)
        top3_local_idx = np.argsort(distances)[:3]
        top3_global_idx = chunk_indices[top3_local_idx]

        representative = []
        for idx in top3_global_idx:
            m = metadata[idx]
            representative.append({
                "chunk_index" : int(idx),
                "title"       : m.get("title", "N/A"),
                "text_preview": m.get("text", "")[:300]
            })

        blind_spot_summaries.append({
            "cluster_id"          : int(cluster_id),
            "n_chunks"            : int(len(chunk_indices)),
            "representative_chunks": representative
        })

    return {
        "n_clusters"          : n_clusters,
        "n_covered"           : len(covered_clusters),
        "n_blind_spots"       : len(blind_spot_clusters),
        "covered_cluster_ids" : sorted(covered_clusters),
        "blind_spot_cluster_ids": sorted(blind_spot_clusters),
        "coverage_rate"       : coverage_rate,
        "coverage_score"      : coverage_score,
        "blind_spot_summaries": blind_spot_summaries,
        "queries_in_noise_territory": int(query_is_noise.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: DIVERSITY SCORE (RQ2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_diversity_score(
    query_embeddings: np.ndarray,
    query_assignments: np.ndarray,
    query_is_noise: np.ndarray,
    query_texts: list,
    query_ids: list,
    n_clusters: int
) -> dict:
    """
    RQ2: Are the questions semantically diverse or redundant?

    Part A — Near-duplicate detection
      Pairwise cosine similarity between all query pairs.
      Pairs above threshold are candidate near-duplicates.
      redundancy_rate = n_near_duplicate_pairs / n_queries

    Part B — Query spread
      How spread out are the queries in the corpus cluster space?
      inter_cluster_spread = mean pairwise cosine distance between
                             cluster centroids (computed from query embeddings)

    Diversity Score = 10 × (1 - redundancy_rate) × normalized_spread

    Returns dict with all RQ2 metrics.
    """
    print(f"\n[6b/7] Computing Diversity Score (RQ2)...")
    n_queries = len(query_texts)

    # ── Part A: Near-duplicate detection ─────────────────────────────────────
    sim_matrix = cosine_similarity(query_embeddings)
    np.fill_diagonal(sim_matrix, 0.0)  # ignore self-similarity

    threshold = CONFIG["near_duplicate_threshold"]
    near_duplicate_pairs = []
    for i in range(n_queries):
        for j in range(i + 1, n_queries):
            sim = float(sim_matrix[i][j])
            if sim >= threshold:
                near_duplicate_pairs.append({
                    "query_1_id"       : query_ids[i],
                    "query_1"          : query_texts[i],
                    "query_2_id"       : query_ids[j],
                    "query_2"          : query_texts[j],
                    "cosine_similarity": sim
                })

    near_duplicate_pairs.sort(key=lambda x: -x["cosine_similarity"])
    n_near_dup      = len(near_duplicate_pairs)
    redundancy_rate = n_near_dup / n_queries

    print(f"      Near-duplicate pairs (>={threshold}): {n_near_dup}")
    for pair in near_duplicate_pairs[:5]:  # Print top 5
        print(f"        sim={pair['cosine_similarity']:.4f} | "
              f"{pair['query_1'][:60]}... | {pair['query_2'][:60]}...")

    # ── Part B: Query spread ──────────────────────────────────────────────────
    # Compute centroid per query cluster (using actual query embeddings)
    assigned_clusters = query_assignments[~query_is_noise]
    cluster_ids_present = sorted(set(assigned_clusters.tolist()))
    n_query_clusters = len(cluster_ids_present)

    if n_query_clusters >= 2:
        centroids = []
        for cid in cluster_ids_present:
            cluster_mask = (query_assignments == cid) & (~query_is_noise)
            cluster_vecs = query_embeddings[cluster_mask]
            centroids.append(cluster_vecs.mean(axis=0))
        centroids = np.array(centroids)

        centroid_distances = pairwise_distances(centroids, metric="cosine")
        np.fill_diagonal(centroid_distances, np.nan)
        mean_inter_cluster_dist = float(np.nanmean(centroid_distances))

        # Normalize spread to [0,1]: 0 = all centroids identical, 1 = max spread
        # Cosine distance is bounded [0,2], so we normalize by 2
        normalized_spread = min(mean_inter_cluster_dist / 2.0, 1.0)
    else:
        mean_inter_cluster_dist = 0.0
        normalized_spread       = 0.0
        print("      Warning: fewer than 2 query clusters found — spread = 0")

    # ── Diversity Score ───────────────────────────────────────────────────────
    diversity_score = 10.0 * (1.0 - redundancy_rate) * normalized_spread

    print(f"      Redundancy rate:              {redundancy_rate:.3f}")
    print(f"      Mean inter-cluster distance:  {mean_inter_cluster_dist:.4f}")
    print(f"      Normalized spread:            {normalized_spread:.4f}")
    print(f"      Diversity score (0-10):       {diversity_score:.2f}")

    # Per-cluster intra-density (for detailed output)
    intra_densities = {}
    for cid in cluster_ids_present:
        cluster_mask = (query_assignments == cid) & (~query_is_noise)
        cluster_vecs = query_embeddings[cluster_mask]
        if cluster_vecs.shape[0] >= 2:
            pw = pairwise_distances(cluster_vecs, metric="cosine")
            np.fill_diagonal(pw, np.nan)
            intra_densities[int(cid)] = float(np.nanmean(pw))
        else:
            intra_densities[int(cid)] = 0.0

    return {
        "n_queries"                  : n_queries,
        "n_near_duplicate_pairs"     : n_near_dup,
        "redundancy_rate"            : redundancy_rate,
        "near_duplicate_pairs"       : near_duplicate_pairs,
        "n_query_clusters"           : n_query_clusters,
        "mean_inter_cluster_distance": mean_inter_cluster_dist,
        "normalized_spread"          : normalized_spread,
        "intra_cluster_densities"    : intra_densities,
        "diversity_score"            : diversity_score,
        "similarity_threshold_used"  : threshold,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7: VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_visualizations(
    corpus_2d        : np.ndarray,
    query_2d         : np.ndarray,
    corpus_labels    : np.ndarray,
    query_assignments: np.ndarray,
    query_is_noise   : np.ndarray,
    coverage_results : dict,
    diversity_results: dict,
    output_dir       : str
):
    """Generate all visualizations and save to output_dir."""
    print(f"\n[7/7] Generating visualizations...")
    os.makedirs(output_dir, exist_ok=True)
    n_clusters = coverage_results["n_clusters"]

    # ── Plot 1: UMAP 2D — Corpus + Query Overlay ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))

    # Corpus chunks — colored by HDBSCAN cluster, noise in gray
    non_noise_mask = corpus_labels >= 0
    noise_mask     = corpus_labels == -1

    if noise_mask.any():
        ax.scatter(
            corpus_2d[noise_mask, 0], corpus_2d[noise_mask, 1],
            c="lightgray", alpha=0.2, s=3, label="Corpus (noise)"
        )

    if non_noise_mask.any() and n_clusters > 0:
        sc = ax.scatter(
            corpus_2d[non_noise_mask, 0], corpus_2d[non_noise_mask, 1],
            c=corpus_labels[non_noise_mask],
            cmap="tab20" if n_clusters <= 20 else "nipy_spectral",
            alpha=0.3, s=3
        )

    # Queries — covered vs. noise territory
    covered_mask = ~query_is_noise
    noise_q_mask = query_is_noise

    if covered_mask.any():
        ax.scatter(
            query_2d[covered_mask, 0], query_2d[covered_mask, 1],
            c="red", s=100, marker="*", zorder=5, alpha=0.9,
            label="Query (assigned)"
        )
    if noise_q_mask.any():
        ax.scatter(
            query_2d[noise_q_mask, 0], query_2d[noise_q_mask, 1],
            c="orange", s=80, marker="*", zorder=5, alpha=0.7,
            label="Query (noise territory)"
        )

    coverage_score = coverage_results["coverage_score"]
    coverage_rate  = coverage_results["coverage_rate"]
    n_blind        = coverage_results["n_blind_spots"]

    ax.set_title(
        f"Corpus Coverage Analysis — UMAP 2D\n"
        f"Coverage Score: {coverage_score:.1f}/10 | "
        f"Rate: {coverage_rate:.1%} | "
        f"Blind spots: {n_blind} clusters",
        fontsize=13
    )
    ax.legend(fontsize=10, markerscale=2, loc="upper right")
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "umap_2d_coverage.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")

    # ── Plot 2: HDBSCAN Coverage Map — Blind Spots Highlighted ───────────────
    fig, ax = plt.subplots(figsize=(14, 9))

    if noise_mask.any():
        ax.scatter(corpus_2d[noise_mask, 0], corpus_2d[noise_mask, 1],
                   c="lightgray", alpha=0.15, s=3)

    if non_noise_mask.any() and n_clusters > 0:
        ax.scatter(
            corpus_2d[non_noise_mask, 0], corpus_2d[non_noise_mask, 1],
            c=corpus_labels[non_noise_mask],
            cmap="tab20" if n_clusters <= 20 else "nipy_spectral",
            alpha=0.25, s=3
        )

    # Queries in covered clusters
    blind_set = set(coverage_results["blind_spot_cluster_ids"])
    for i in range(len(query_assignments)):
        if query_is_noise[i]:
            color, marker, zorder = "orange", "*", 5
        elif int(query_assignments[i]) in blind_set:
            color, marker, zorder = "blue", "D", 6   # shouldn't happen but guard
        else:
            color, marker, zorder = "red", "*", 5
        ax.scatter(query_2d[i, 0], query_2d[i, 1],
                   c=color, s=100, marker=marker, zorder=zorder, alpha=0.85)

    # Legend
    legend_elements = [
        mlines.Line2D([0], [0], marker="*", color="w",
                      markerfacecolor="red", markersize=12,
                      label="Query — covered cluster"),
        mlines.Line2D([0], [0], marker="*", color="w",
                      markerfacecolor="orange", markersize=12,
                      label="Query — noise territory"),
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc="upper right")
    ax.set_title(
        f"HDBSCAN Coverage Map\n"
        f"{n_clusters} clusters | "
        f"Coverage: {coverage_rate:.1%} | "
        f"Blind spots: {n_blind}",
        fontsize=13
    )
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "hdbscan_coverage_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")

    # ── Plot 3: Query Spread Map (RQ2) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 7))

    ax.scatter(
        query_2d[:, 0], query_2d[:, 1],
        c=query_assignments, cmap="tab10",
        s=120, alpha=0.85, zorder=3
    )

    # Annotate each query with its index
    for i, (x, y) in enumerate(query_2d):
        ax.annotate(str(i), (x, y), fontsize=6, alpha=0.5,
                    xytext=(3, 3), textcoords="offset points")

    diversity_score = diversity_results["diversity_score"]
    n_dup           = diversity_results["n_near_duplicate_pairs"]
    n_q_clusters    = diversity_results["n_query_clusters"]

    ax.set_title(
        f"Query Diversity Analysis — UMAP 2D\n"
        f"Diversity Score: {diversity_score:.1f}/10 | "
        f"Near-duplicates: {n_dup} pairs | "
        f"Query clusters: {n_q_clusters}",
        fontsize=13
    )
    ax.set_xlabel("UMAP Dim 1")
    ax.set_ylabel("UMAP Dim 2")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "query_spread_map.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────────────────────

def generate_recommendations(
    coverage_results: dict,
    diversity_results: dict
) -> list:
    """
    Threshold-triggered recommendations based on scores.
    Returns a list of recommendation strings.
    """
    recs = []
    cov_score = coverage_results["coverage_score"]
    div_score = diversity_results["diversity_score"]
    cov_rate  = coverage_results["coverage_rate"]
    n_blind   = coverage_results["n_blind_spots"]
    n_dup     = diversity_results["n_near_duplicate_pairs"]
    n_q       = diversity_results["n_queries"]

    # Coverage recommendations
    good_t    = CONFIG["coverage_thresholds"]["good"]
    warn_t    = CONFIG["coverage_thresholds"]["warning"]

    if cov_rate >= good_t:
        recs.append(
            f"✅ COVERAGE: Good ({cov_rate:.1%}). "
            f"Question bank covers {cov_rate:.1%} of corpus clusters. No action required."
        )
    elif cov_rate >= warn_t:
        recs.append(
            f"⚠️  COVERAGE: Moderate ({cov_rate:.1%}). "
            f"{n_blind} corpus clusters have no questions. "
            f"Review blind_spot_clusters.json and add questions targeting "
            f"uncovered topics."
        )
    else:
        recs.append(
            f"❌ COVERAGE: Poor ({cov_rate:.1%}). "
            f"{n_blind} of {coverage_results['n_clusters']} corpus clusters "
            f"have no questions. The question bank covers less than half the "
            f"knowledge base. Significant expansion required. See "
            f"blind_spot_clusters.json for uncovered topics."
        )

    # Diversity recommendations
    good_d = CONFIG["diversity_thresholds"]["good"]
    warn_d = CONFIG["diversity_thresholds"]["warning"]

    if div_score >= good_d:
        recs.append(
            f"✅ DIVERSITY: Good ({div_score:.1f}/10). "
            f"Questions are semantically diverse with {n_dup} near-duplicate pairs."
        )
    elif div_score >= warn_d:
        recs.append(
            f"⚠️  DIVERSITY: Moderate ({div_score:.1f}/10). "
            f"{n_dup} near-duplicate pairs detected. "
            f"Consider replacing redundant questions with ones covering "
            f"underrepresented corpus regions."
        )
    else:
        recs.append(
            f"❌ DIVERSITY: Poor ({div_score:.1f}/10). "
            f"{n_dup} near-duplicate pairs out of {n_q} questions — "
            f"effective unique questions may be substantially fewer than {n_q}. "
            f"Remove or rephrase near-duplicates. See near_duplicate_pairs.json."
        )

    # Noise territory
    n_noise_q = coverage_results["queries_in_noise_territory"]
    if n_noise_q > 0:
        recs.append(
            f"ℹ️  NOISE TERRITORY: {n_noise_q} queries do not strongly belong "
            f"to any corpus cluster. These may be out-of-domain questions or "
            f"cover very sparse corpus regions. Review manually."
        )

    return recs


# ─────────────────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    coverage_results : dict,
    diversity_results: dict,
    recommendations  : list,
    output_dir       : str
):
    """Save all JSON outputs and the human-readable diagnostic report."""
    os.makedirs(output_dir, exist_ok=True)

    # ── diagnostic_report.json ────────────────────────────────────────────────
    report = {
        "generated_at"    : datetime.utcnow().isoformat(),
        "config"          : CONFIG,
        "coverage_score"  : coverage_results["coverage_score"],
        "diversity_score" : diversity_results["diversity_score"],
        "rq1_coverage"    : {
            k: v for k, v in coverage_results.items()
            if k != "blind_spot_summaries"
        },
        "rq2_diversity"   : {
            k: v for k, v in diversity_results.items()
            if k != "near_duplicate_pairs"
        },
        "recommendations" : recommendations,
    }
    with open(os.path.join(output_dir, "diagnostic_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ── blind_spot_clusters.json ──────────────────────────────────────────────
    with open(os.path.join(output_dir, "blind_spot_clusters.json"), "w") as f:
        json.dump(coverage_results["blind_spot_summaries"], f, indent=2)

    # ── near_duplicate_pairs.json ─────────────────────────────────────────────
    with open(os.path.join(output_dir, "near_duplicate_pairs.json"), "w") as f:
        json.dump(diversity_results["near_duplicate_pairs"], f, indent=2)

    # ── diagnostic_report.txt — human-readable ────────────────────────────────
    lines = [
        "=" * 60,
        "  QUESTION BANK DIAGNOSTIC REPORT",
        "  Semantic Coverage Analysis — Stanford CS321M",
        f"  Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "=" * 60,
        "",
        "COVERAGE AUDIT (RQ1: Corpus Coverage)",
        "-" * 40,
        f"  Coverage Score:      {coverage_results['coverage_score']:.1f} / 10",
        f"  Coverage Rate:       {coverage_results['coverage_rate']:.1%}",
        f"  Total clusters:      {coverage_results['n_clusters']}",
        f"  Covered clusters:    {coverage_results['n_covered']}",
        f"  Blind spot clusters: {coverage_results['n_blind_spots']}",
        f"  Queries in noise:    {coverage_results['queries_in_noise_territory']}",
        "",
        "DIVERSITY AUDIT (RQ2: Query Redundancy)",
        "-" * 40,
        f"  Diversity Score:          {diversity_results['diversity_score']:.1f} / 10",
        f"  Near-duplicate pairs:     {diversity_results['n_near_duplicate_pairs']}",
        f"  Redundancy rate:          {diversity_results['redundancy_rate']:.3f}",
        f"  Query clusters found:     {diversity_results['n_query_clusters']}",
        f"  Mean inter-cluster dist:  {diversity_results['mean_inter_cluster_distance']:.4f}",
        f"  Normalized spread:        {diversity_results['normalized_spread']:.4f}",
        f"  Similarity threshold:     {diversity_results['similarity_threshold_used']}",
        "",
        "RECOMMENDATIONS",
        "-" * 40,
    ]
    for rec in recommendations:
        lines.append(f"  {rec}")
    lines += [
        "",
        "OUTPUT FILES",
        "-" * 40,
        "  diagnostic_report.json   — all metrics",
        "  blind_spot_clusters.json — uncovered corpus clusters",
        "  near_duplicate_pairs.json— near-duplicate query pairs",
        "  umap_2d_coverage.png     — corpus + query overlay",
        "  hdbscan_coverage_map.png — cluster coverage map",
        "  query_spread_map.png     — query diversity map",
        "=" * 60,
    ]

    txt_path = os.path.join(output_dir, "diagnostic_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n      All outputs saved to: {output_dir}/")
    print(f"      diagnostic_report.txt")
    print(f"      diagnostic_report.json")
    print(f"      blind_spot_clusters.json")
    print(f"      near_duplicate_pairs.json")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    corpus_embeddings_path : str,
    questions_path         : str,
    metadata_path          : str,
    embedder_name_path     : str,
    output_dir             : str
):
    """
    End-to-end diagnostic pipeline.
    Inputs  → corpus embeddings + query JSON
    Outputs → coverage score + diversity score + report + visualizations
    """
    print("\n" + "=" * 60)
    print("  SEMANTIC COVERAGE DIAGNOSTIC")
    print("  Stanford CS321M — Phase 1 Question Bank Audit")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load inputs
    corpus_embeddings = load_corpus_embeddings(corpus_embeddings_path)
    metadata          = load_corpus_metadata(metadata_path)
    questions         = load_questions(questions_path)

    # 2. Embed queries
    query_embeddings, query_texts, query_ids = embed_queries(
        questions, embedder_name_path
    )

    # Validate embedding dimensions match
    assert corpus_embeddings.shape[1] == query_embeddings.shape[1], (
        f"Dimension mismatch: corpus={corpus_embeddings.shape[1]}, "
        f"queries={query_embeddings.shape[1]}. "
        f"Must use the same embedder for both."
    )

    # 3. UMAP reductions
    corpus_2d, query_2d = run_umap_2d(corpus_embeddings, query_embeddings)
    corpus_hd, query_hd = run_umap_hd(corpus_embeddings, query_embeddings)

    # 4. HDBSCAN clustering
    clusterer = run_hdbscan(corpus_hd)

    # 5. Assign queries to clusters
    query_assignments, query_is_noise, soft_clusters = assign_queries_to_clusters(
        clusterer, query_hd
    )

    # 6a. Coverage score (RQ1)
    coverage_results = compute_coverage_score(
        clusterer, query_assignments, query_is_noise,
        corpus_embeddings, metadata
    )

    # 6b. Diversity score (RQ2)
    diversity_results = compute_diversity_score(
        query_embeddings, query_assignments, query_is_noise,
        query_texts, query_ids,
        n_clusters=coverage_results["n_clusters"]
    )

    # 7. Visualizations
    generate_visualizations(
        corpus_2d, query_2d,
        clusterer.labels_, query_assignments, query_is_noise,
        coverage_results, diversity_results,
        output_dir
    )

    # Generate recommendations
    recommendations = generate_recommendations(coverage_results, diversity_results)

    # Save all outputs
    save_outputs(coverage_results, diversity_results, recommendations, output_dir)

    # Print final report
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Coverage Score:  {coverage_results['coverage_score']:.1f} / 10")
    print(f"  Diversity Score: {diversity_results['diversity_score']:.1f} / 10")
    print()
    for rec in recommendations:
        print(f"  {rec}")
    print("=" * 60)

    return coverage_results, diversity_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Semantic Coverage Diagnostic for RAG Question Banks"
    )
    parser.add_argument(
        "--corpus_embeddings",
        default="outputs/phase1/coverage/corpus_embeddings.npy",
        help="Path to pre-computed corpus chunk embeddings (.npy)"
    )
    parser.add_argument(
        "--questions",
        default="data/processed/subsets/openrag_text_only_100.json",
        help="Path to question bank JSON file"
    )
    parser.add_argument(
        "--metadata",
        default="data/indexes/dense/metadata.jsonl",
        help="Path to corpus chunk metadata (.jsonl)"
    )
    parser.add_argument(
        "--embedder_name",
        default="data/indexes/dense/embedder_name.txt",
        help="Path to file containing embedder model name"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/phase1/coverage",
        help="Directory for all outputs"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Check dependencies
    missing = []
    if not UMAP_AVAILABLE:
        missing.append("umap-learn")
    if not HDBSCAN_AVAILABLE:
        missing.append("hdbscan")
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        exit(1)

    args = parse_args()

    run_pipeline(
        corpus_embeddings_path = args.corpus_embeddings,
        questions_path         = args.questions,
        metadata_path          = args.metadata,
        embedder_name_path     = args.embedder_name,
        output_dir             = args.output_dir,
    )
