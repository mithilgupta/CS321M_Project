#!/bin/bash
# run_all_systems.sh
# Runs all 12 RAG system configurations sequentially
# Usage: bash scripts/run_all_systems.sh
# Optional: bash scripts/run_all_systems.sh --limit 5

set -e

LIMIT_FLAG=""
if [ ! -z "$2" ]; then
    LIMIT_FLAG="--limit $2"
fi

echo "=================================================="
echo "  RAG BATCH RUNNER — 12 System Configurations"
echo "  Stanford CS321M — Phase 1"
echo "=================================================="
echo ""

GENERATORS=("strong_api" "mid_api" "local_open")
RETRIEVERS=("dense" "hybrid")
SCAFFOLDS=("topk3" "topk8")

TOTAL=12
DONE=0
SKIPPED=0
FAILED=0

echo "Total systems to run: $TOTAL"
echo ""

for g in "${GENERATORS[@]}"; do
    for r in "${RETRIEVERS[@]}"; do
        for s in "${SCAFFOLDS[@]}"; do
            SYSTEM_ID="${g}__${r}__${s}"
            OUT_FILE="outputs/runs/phase_1_1/${SYSTEM_ID}.jsonl"

            echo "--------------------------------------------------"
            echo "  System : $SYSTEM_ID"
            echo "  Output : $OUT_FILE"

            if [ -f "$OUT_FILE" ]; then
                LINE_COUNT=$(wc -l < "$OUT_FILE" | tr -d ' ')
                if [ "$LINE_COUNT" -ge 100 ]; then
                    echo "  Status : SKIPPING — already complete ($LINE_COUNT lines)"
                    SKIPPED=$((SKIPPED + 1))
                    continue
                else
                    echo "  Status : RESUMING — only $LINE_COUNT lines found"
                fi
            fi

            echo "  Status : RUNNING..."
            START=$(date +%s)

            if python -m src.runners.run_single_config \
                --generator "$g" \
                --retriever "$r" \
                --scaffold "$s" \
                $LIMIT_FLAG; then
                END=$(date +%s)
                ELAPSED=$((END - START))
                echo "  Status : DONE in ${ELAPSED}s"
                DONE=$((DONE + 1))
            else
                echo "  Status : FAILED"
                FAILED=$((FAILED + 1))
            fi

            echo ""
        done
    done
done

echo "=================================================="
echo "  BATCH RUN COMPLETE"
echo "  Done    : $DONE"
echo "  Skipped : $SKIPPED"
echo "  Failed  : $FAILED"
echo "  Total   : $TOTAL"
echo "=================================================="
