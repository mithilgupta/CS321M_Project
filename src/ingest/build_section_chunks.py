# import json
# from pathlib import Path
# from tqdm import tqdm

# root = Path("data/raw/open_rag_bench")
# corpus_dirs = list(root.rglob("corpus"))

# if not corpus_dirs:
#     raise FileNotFoundError("Could not find a corpus directory under data/raw/open_rag_bench")

# corpus_dir = corpus_dirs[0]
# out_path = Path("data/processed/corpus/section_chunks.jsonl")
# out_path.parent.mkdir(parents=True, exist_ok=True)

# json_files = list(corpus_dir.glob("*.json"))
# if not json_files:
#     json_files = list(corpus_dir.rglob("*.json"))

# count = 0

# with open(out_path, "w") as out_f:
#     for fp in tqdm(json_files, desc="Processing corpus files"):
#         with open(fp, "r") as f:
#             doc = json.load(f)

#         doc_id = doc["id"]
#         title = doc.get("title", "")
#         abstract = doc.get("abstract", "")
#         authors = doc.get("authors", [])
#         categories = doc.get("categories", [])
#         published = doc.get("published", "")

#         sections = doc.get("sections", [])
#         for i, sec in enumerate(sections):
#             text = sec.get("text", "").strip()
#             if not text:
#                 continue

#             record = {
#                 "chunk_id": f"{doc_id}::sec::{i}",
#                 "doc_id": doc_id,
#                 "section_id": i,
#                 "title": title,
#                 "abstract": abstract,
#                 "authors": authors,
#                 "categories": categories,
#                 "published": published,
#                 "text": text
#             }
#             out_f.write(json.dumps(record) + "\n")
#             count += 1

# print(f"Wrote {count} section chunks to {out_path}")


import json
from pathlib import Path
from tqdm import tqdm

root = Path("data/raw/open_rag_bench")

# Ignore any corpus dirs inside .cache
corpus_dirs = [d for d in root.rglob("corpus") if ".cache" not in str(d)]


print("Found corpus dirs:")
for d in corpus_dirs:
    print(" ", d)

if not corpus_dirs:
    raise FileNotFoundError("Could not find a corpus directory under data/raw/open_rag_bench")

corpus_dir = corpus_dirs[0]
print("\nUsing corpus_dir:", corpus_dir)

out_path = Path("data/processed/corpus/section_chunks.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)

json_files = list(corpus_dir.glob("*.json"))
print("Number of *.json files directly under corpus_dir:", len(json_files))

if not json_files:
    json_files = list(corpus_dir.rglob("*.json"))
    print("Number of *.json files under corpus_dir recursively:", len(json_files))

count = 0

with open(out_path, "w") as out_f:
    for fp in tqdm(json_files, desc="Processing corpus files"):
        print("Processing:", fp)
        with open(fp, "r") as f:
            doc = json.load(f)

        doc_id = doc["id"]
        title = doc.get("title", "")
        abstract = doc.get("abstract", "")
        authors = doc.get("authors", [])
        categories = doc.get("categories", [])
        published = doc.get("published", "")

        sections = doc.get("sections", [])
        for i, sec in enumerate(sections):
            text = sec.get("text", "").strip()
            if not text:
                continue

            record = {
                "chunk_id": f"{doc_id}::sec::{i}",
                "doc_id": doc_id,
                "section_id": i,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "categories": categories,
                "published": published,
                "text": text
            }
            out_f.write(json.dumps(record) + "\n")
            count += 1

print(f"\nWrote {count} section chunks to {out_path}")