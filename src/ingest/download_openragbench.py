from huggingface_hub import snapshot_download

repo_id = "vectara/open_ragbench"
local_dir = "data/raw/open_rag_bench"

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)

print(f"Downloaded dataset to {local_dir}")