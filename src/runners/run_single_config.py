import argparse
import json
from pathlib import Path
import yaml
from tqdm import tqdm
from src.rag.retrieve import DenseRetriever, HybridRetriever
from src.rag.generate import AnthropicGenerator, HFLocalGenerator, build_prompt

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--scaffold",  required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    models_cfg    = load_yaml("configs/models.yaml")["generators"]
    retrievers_cfg= load_yaml("configs/retrievers.yaml")["retrievers"]
    scaffolds_cfg = load_yaml("configs/scaffolds.yaml")["scaffolds"]
    exp_cfg       = load_yaml("configs/experiment_phase_1_1.yaml")

    subset_path = exp_cfg["question_subset"]
    with open(subset_path, "r") as f:
        questions = json.load(f)
    if args.limit is not None:
        questions = questions[:args.limit]

    model_cfg    = models_cfg[args.generator]
    retr_cfg     = retrievers_cfg[args.retriever]
    scaffold_cfg = scaffolds_cfg[args.scaffold]

    # Build retriever
    if retr_cfg["type"] == "dense":
        retriever = DenseRetriever()
    elif retr_cfg["type"] == "hybrid":
        retriever = HybridRetriever(
            alpha_dense=retr_cfg.get("alpha_dense", 0.5),
            alpha_bm25=retr_cfg.get("alpha_bm25", 0.5)
        )
    else:
        raise ValueError(f"Unknown retriever type: {retr_cfg['type']}")

    # Build generator
    if model_cfg["provider"] == "anthropic":
        generator = AnthropicGenerator(model_cfg["model_name"])
    elif model_cfg["provider"] == "hf_local":
        generator = HFLocalGenerator(model_cfg["model_name"])
    else:
        raise ValueError(f"Unknown provider: {model_cfg['provider']}")

    top_k     = scaffold_cfg["top_k"]
    out_dir   = Path("outputs/runs/phase_1_1")
    out_dir.mkdir(parents=True, exist_ok=True)
    system_id = f"{args.generator}__{args.retriever}__{args.scaffold}"
    out_path  = out_dir / f"{system_id}.jsonl"

    # Skip if already complete
    if out_path.exists():
        with open(out_path) as f:
            existing = sum(1 for _ in f)
        if existing >= len(questions):
            print(f"Already complete: {out_path} ({existing} records). Skipping.")
            return

    with open(out_path, "w") as out_f:
        for q in tqdm(questions, desc=system_id):
            question  = q["question"]
            retrieved = retriever.search(question, top_k=top_k)
            prompt    = build_prompt(question, retrieved)
            try:
                answer = generator.generate(prompt)
                status = "success"
                error  = None
            except Exception as e:
                answer = ""
                status = "error"
                error  = str(e)

            record = {
                "question_id"      : q["question_id"],
                "question"         : q["question"],
                "question_type"    : q["question_type"],
                "source"           : q["source"],
                "gold_doc_id"      : q["gold_doc_id"],
                "gold_section_id"  : q["gold_section_id"],
                "reference_answer" : q["reference_answer"],
                "system_id"        : system_id,
                "generator_key"    : args.generator,
                "generator_model"  : model_cfg["model_name"],
                "retriever_key"    : args.retriever,
                "retriever_type"   : retr_cfg["type"],
                "scaffold_key"     : args.scaffold,
                "top_k"            : top_k,
                "retrieved_chunks" : retrieved,
                "generated_answer" : answer,
                "run_status"       : status,
                "error"            : error
            }
            out_f.write(json.dumps(record) + "\n")

    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
