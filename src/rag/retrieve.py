import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, index_dir="data/indexes/dense"):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "faiss.index"))

        with open(index_dir / "metadata.jsonl", "r") as f:
            self.metadata = [json.loads(line) for line in f]

        with open(index_dir / "embedder_name.txt", "r") as f:
            model_name = f.read().strip()

        self.embedder = SentenceTransformer(model_name)

    def search(self, query, top_k=5):
        q_emb = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            rec = self.metadata[idx]
            results.append({
                "score": float(score),
                **rec
            })
        return results


class HybridRetriever:
    def __init__(self, dense_index_dir="data/indexes/dense", hybrid_dir="data/indexes/hybrid", alpha_dense=0.5, alpha_bm25=0.5):
        self.dense = DenseRetriever(dense_index_dir)
        self.alpha_dense = alpha_dense
        self.alpha_bm25 = alpha_bm25

        with open(Path(hybrid_dir) / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        with open(Path(hybrid_dir) / "metadata.jsonl", "r") as f:
            self.metadata = [json.loads(line) for line in f]

    def search(self, query, top_k=5, bm25_pool=50, dense_pool=50):
        dense_results = self.dense.search(query, top_k=dense_pool)
        dense_scores = {r["chunk_id"]: r["score"] for r in dense_results}

        query_tokens = query.lower().split()
        bm25_scores_arr = self.bm25.get_scores(query_tokens)

        bm25_top_idx = np.argsort(bm25_scores_arr)[::-1][:bm25_pool]
        bm25_scores = {self.metadata[idx]["chunk_id"]: float(bm25_scores_arr[idx]) for idx in bm25_top_idx}

        all_chunk_ids = set(dense_scores.keys()) | set(bm25_scores.keys())

        if dense_scores:
            dense_max = max(dense_scores.values())
        else:
            dense_max = 1.0

        if bm25_scores:
            bm25_max = max(bm25_scores.values())
        else:
            bm25_max = 1.0

        fused = []
        by_id = {rec["chunk_id"]: rec for rec in self.metadata}

        for cid in all_chunk_ids:
            d = dense_scores.get(cid, 0.0) / (dense_max if dense_max != 0 else 1.0)
            b = bm25_scores.get(cid, 0.0) / (bm25_max if bm25_max != 0 else 1.0)
            score = self.alpha_dense * d + self.alpha_bm25 * b
            fused.append((score, by_id[cid]))

        fused.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, rec in fused[:top_k]:
            results.append({
                "score": float(score),
                **rec
            })
        return results