#!/bin/bash
# Benchmark TurboQuant vs q8_0/q4_0 on M5 Max
# Usage: bash benchmarks/benchmark_llama.sh
#
# Copyright 2026 Tom Turney. Licensed under Apache 2.0.

LLAMA_DIR="$HOME/local_llms/llama.cpp"
MODEL="${MODEL:-$HOME/local_llms/models/Qwen3.5-27B-Q5_K_M.gguf}"
CLI="$LLAMA_DIR/build-turbo/bin/llama-cli"
RESULTS_FILE="benchmarks/benchmark_results.md"

# Multi-GPU tensor split (RTX 5060 Ti 16GB + RTX 3060 12GB with ~9GB free after monitors)
# Override via env: TENSOR_SPLIT=0.63,0.37 bash benchmark_llama.sh
TENSOR_SPLIT="${TENSOR_SPLIT:-0.63,0.37}"

echo "=============================================="
echo "TurboQuant Benchmark — RTX 5060 Ti + RTX 3060"
echo "Model: $(basename $MODEL)"
echo "Tensor split: $TENSOR_SPLIT"
echo "=============================================="
echo ""

echo "| Cache Type | Bits/val | Prompt tok/s | Gen tok/s | KV Compression |" > "$RESULTS_FILE"
echo "|------------|----------|-------------|-----------|----------------|" >> "$RESULTS_FILE"

for CACHE_TYPE in "q8_0" "q4_0" "turbo3" "turbo4"; do
    echo "--- Testing $CACHE_TYPE ---"

    # Run with --perf and capture timing lines
    OUTPUT=$($CLI \
        -m "$MODEL" \
        -ngl 99 -fa on \
        --tensor-split "$TENSOR_SPLIT" \
        -c 2048 \
        -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" \
        -n 50 -p "Explain quantum computing in simple terms." \
        --no-display-prompt --jinja \
        2>&1)

    # Extract timing info
    PP=$(echo "$OUTPUT" | grep "prompt eval" | grep -oE '[0-9]+\.[0-9]+ tokens per second' | grep -oE '[0-9]+\.[0-9]+')
    TG=$(echo "$OUTPUT" | grep "eval time\|generation" | grep -oE '[0-9]+\.[0-9]+ tokens per second' | grep -oE '[0-9]+\.[0-9]+')

    # Compression ratio
    case $CACHE_TYPE in
        q8_0)   RATIO="2.0×" ; BITS="8" ;;
        q4_0)   RATIO="4.0×" ; BITS="4" ;;
        turbo3) RATIO="4.9×" ; BITS="3.25" ;;
        turbo4) RATIO="3.8×" ; BITS="4.25" ;;
    esac

    echo "  Prompt: ${PP:-N/A} tok/s, Gen: ${TG:-N/A} tok/s"
    echo "| $CACHE_TYPE | $BITS | ${PP:-N/A} | ${TG:-N/A} | $RATIO |" >> "$RESULTS_FILE"
done

echo ""
echo "Results saved to $RESULTS_FILE"
cat "$RESULTS_FILE"
