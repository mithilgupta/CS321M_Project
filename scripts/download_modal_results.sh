#!/bin/bash
VOLUME="rag-eval-outputs"
LOCAL_DIR="outputs/runs/phase_1_1"
mkdir -p "$LOCAL_DIR"

SYSTEMS=(
    "strong_api__dense__topk3"
    "strong_api__dense__topk8"
    "strong_api__hybrid__topk3"
    "strong_api__hybrid__topk8"
    "mid_api__dense__topk3"
    "mid_api__dense__topk8"
    "mid_api__hybrid__topk3"
    "mid_api__hybrid__topk8"
    "local_open__dense__topk3"
    "local_open__dense__topk8"
    "local_open__hybrid__topk3"
    "local_open__hybrid__topk8"
)

for sid in "${SYSTEMS[@]}"; do
    FILE="${sid}.jsonl"
    echo "Downloading: $FILE"
    modal volume get --force "$VOLUME" "$FILE" "$LOCAL_DIR/$FILE" && \
        echo "  ✅ $FILE" || \
        echo "  ❌ $FILE"
done

echo ""
echo "Done. Line counts:"
wc -l "$LOCAL_DIR"/*.jsonl
