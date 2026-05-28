import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm

chunks_path = Path("data/processed/corpus/section_chunks.jsonl")
out_dir = Path("data/indexes/hybrid")
out_dir.mkdir(parents=True, exist_ok=True)

metadata = []
tokenized_corpus = []

with open(chunks_path, "r") as f:
    for line in tqdm(f, desc="Loading chunks"):
        rec = json.loads(line)
        text = f"{rec['title']} {rec['abstract']} {rec['text']}".lower()
        tokens = text.split()
        tokenized_corpus.append(tokens)
        metadata.append(rec)

bm25 = BM25Okapi(tokenized_corpus)

with open(out_dir / "bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

with open(out_dir / "metadata.jsonl", "w") as f:
    for rec in metadata:
        f.write(json.dumps(rec) + "\n")

print("BM25 index built.")
print("Chunks indexed:", len(metadata))