import json
import random
from pathlib import Path

random.seed(42)

root = Path("data/raw/open_rag_bench")
queries_path = list(root.rglob("queries.json"))[0]
qrels_path = list(root.rglob("qrels.json"))[0]
answers_path = list(root.rglob("answers.json"))[0]

with open(queries_path, "r") as f:
    queries = json.load(f)
with open(qrels_path, "r") as f:
    qrels = json.load(f)
with open(answers_path, "r") as f:
    answers = json.load(f)

text_only = []
for qid, qobj in queries.items():
    if qobj.get("source") == "text":
        if qid in qrels and qid in answers:
            text_only.append((qid, qobj))

extractive = [(qid, qobj) for qid, qobj in text_only if qobj.get("type") == "extractive"]
abstractive = [(qid, qobj) for qid, qobj in text_only if qobj.get("type") == "abstractive"]

random.shuffle(extractive)
random.shuffle(abstractive)

target_each = 50
subset = extractive[:target_each] + abstractive[:target_each]
random.shuffle(subset)

records = []
for qid, qobj in subset:
    qrel = qrels[qid]
    rec = {
        "question_id": qid,
        "question": qobj["query"],
        "question_type": qobj["type"],
        "source": qobj["source"],
        "gold_doc_id": qrel["doc_id"],
        "gold_section_id": qrel["section_id"],
        "reference_answer": answers[qid]
    }
    records.append(rec)

out_path = Path("data/processed/subsets/openrag_text_only_100.json")
with open(out_path, "w") as f:
    json.dump(records, f, indent=2)

print(f"Wrote {len(records)} questions to {out_path}")
print("Extractive:", sum(r["question_type"] == "extractive" for r in records))
print("Abstractive:", sum(r["question_type"] == "abstractive" for r in records))