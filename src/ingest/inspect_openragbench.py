import os
import json
from pathlib import Path

root = Path("data/raw/open_rag_bench")

files_to_find = ["queries.json", "qrels.json", "answers.json", "pdf_urls.json"]
found = {}

for name in files_to_find:
    matches = list(root.rglob(name))
    found[name] = matches

for name, matches in found.items():
    print(f"\n{name}")
    for m in matches:
        print("  ", m)

corpus_dirs = list(root.rglob("corpus"))
print("\ncorpus directories:")
for d in corpus_dirs:
    print("  ", d)

queries_path = found["queries.json"][0]
qrels_path = found["qrels.json"][0]
answers_path = found["answers.json"][0]

with open(queries_path, "r") as f:
    queries = json.load(f)
with open(qrels_path, "r") as f:
    qrels = json.load(f)
with open(answers_path, "r") as f:
    answers = json.load(f)

print("\nCounts")
print("queries:", len(queries))
print("qrels:", len(qrels))
print("answers:", len(answers))

first_qid = next(iter(queries))
print("\nSample query id:", first_qid)
print("Sample query object:", json.dumps(queries[first_qid], indent=2)[:1000])
print("Sample qrel object:", json.dumps(qrels[first_qid], indent=2)[:1000])
print("Sample answer:", str(answers[first_qid])[:1000])