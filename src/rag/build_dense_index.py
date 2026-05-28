import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss

chunks_path = Path("data/processed/corpus/section_chunks.jsonl")
index_dir = Path("data/indexes/dense")
index_dir.mkdir(parents=True, exist_ok=True)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

texts = []
metadata = []

with open(chunks_path, "r") as f:
    for line in f:
        rec = json.loads(line)
        text = f"Title: {rec['title']}\n\nAbstract: {rec['abstract']}\n\nSection: {rec['text']}"
        texts.append(text)
        metadata.append(rec)

embeddings = []
batch_size = 64

for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
    batch = texts[i:i+batch_size]
    embs = embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    embeddings.append(embs)

embeddings = np.vstack(embeddings).astype("float32")

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, str(index_dir / "faiss.index"))

with open(index_dir / "metadata.jsonl", "w") as f:
    for rec in metadata:
        f.write(json.dumps(rec) + "\n")

with open(index_dir / "embedder_name.txt", "w") as f:
    f.write(model_name)

print("Dense index built.")
print("Chunks indexed:", len(metadata))
print("Embedding dim:", embeddings.shape[1])