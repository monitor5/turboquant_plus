#!/usr/bin/env bash
# launch_server.sh — Multi-GPU llama-server launcher with VRAM-aware tensor split
#
# Strategy:
#   1. Query each GPU's total VRAM via nvidia-smi.
#   2. Reserve VRAM_RESERVE_PCT (default 3%) headroom on every GPU.
#   3. Sort GPUs by usable VRAM descending (biggest GPU is primary / main-gpu).
#   4. Pass --tensor-split proportional to usable VRAM so llama.cpp distributes
#      layers accordingly — the biggest GPU absorbs the most weight.
#
# Usage:
#   ./scripts/launch_server.sh -m /path/to/model.gguf [options]
#
# Extra options are forwarded verbatim to llama-server, e.g.:
#   ./scripts/launch_server.sh -m model.gguf -c 32768 --port 8081
#
# Env overrides:
#   LLAMA_DIR         — path to built llama.cpp directory (default: ../llama-cpp-turboquant)
#   VRAM_RESERVE_PCT  — % of VRAM to keep free per GPU        (default: 3)
#   CACHE_TYPE        — turbo3 | turbo4 | q8_0 | q4_0         (default: turbo4)
#   CTX_SIZE          — context window size                    (default: 32768)
#   PORT              — server port                            (default: 8080)
#   NGL               — GPU layers to offload (999 = all)      (default: 999)

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (overridable via env)
# ---------------------------------------------------------------------------
LLAMA_DIR="${LLAMA_DIR:-$(dirname "$0")/../../llama-cpp-turboquant}"
VRAM_RESERVE_PCT="${VRAM_RESERVE_PCT:-3}"
CACHE_TYPE="${CACHE_TYPE:-turbo4}"
CTX_SIZE="${CTX_SIZE:-32768}"
PORT="${PORT:-8080}"
NGL="${NGL:-999}"

# ---------------------------------------------------------------------------
# Parse -m / --model; forward everything else to llama-server
# ---------------------------------------------------------------------------
MODEL_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL_PATH="$2"; shift 2 ;;
        -c|--ctx-size)
            CTX_SIZE="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --cache-type-k|--cache-type-v)
            # Let user override individually — store and forward
            EXTRA_ARGS+=("$1" "$2"); shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 -m /path/to/model.gguf [extra llama-server args]"
    exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: model file not found: $MODEL_PATH"
    exit 1
fi

# ---------------------------------------------------------------------------
# Locate llama-server binary
# ---------------------------------------------------------------------------
LLAMA_BIN=""
for candidate in \
    "${LLAMA_DIR}/build/bin/llama-server" \
    "${LLAMA_DIR}/llama-server" \
    "$(command -v llama-server 2>/dev/null || true)"; do
    if [[ -x "$candidate" ]]; then
        LLAMA_BIN="$candidate"
        break
    fi
done

if [[ -z "$LLAMA_BIN" ]]; then
    echo "ERROR: llama-server not found."
    echo "  Build it with: cmake -B build -DGGML_CUDA=ON && cmake --build build -j"
    echo "  or set LLAMA_DIR=/path/to/llama-cpp-turboquant"
    exit 1
fi

# ---------------------------------------------------------------------------
# Query GPUs and compute usable VRAM
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

# Arrays: gpu_ids, gpu_names, usable_mib
declare -a GPU_IDS GPU_NAMES USABLE_MIB

while IFS=',' read -r idx name total_raw; do
    idx="${idx// /}"
    name="${name## }"
    total_mib="${total_raw// MiB/}"; total_mib="${total_mib// /}"
    reserve=$(( total_mib * VRAM_RESERVE_PCT / 100 ))
    usable=$(( total_mib - reserve ))
    GPU_IDS+=("$idx")
    GPU_NAMES+=("$name")
    USABLE_MIB+=("$usable")
done < <(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)

GPU_COUNT=${#GPU_IDS[@]}

if [[ $GPU_COUNT -eq 0 ]]; then
    echo "ERROR: No NVIDIA GPUs detected."
    exit 1
fi

# ---------------------------------------------------------------------------
# Sort by usable VRAM descending → pick main GPU (biggest)
# ---------------------------------------------------------------------------
# Build a sorted order (simple bubble-like using index array)
declare -a SORTED_ORDER
for (( i=0; i<GPU_COUNT; i++ )); do SORTED_ORDER+=("$i"); done

for (( i=0; i<GPU_COUNT-1; i++ )); do
    for (( j=i+1; j<GPU_COUNT; j++ )); do
        ai="${SORTED_ORDER[$i]}"; aj="${SORTED_ORDER[$j]}"
        if (( USABLE_MIB[aj] > USABLE_MIB[ai] )); then
            SORTED_ORDER[$i]="$aj"; SORTED_ORDER[$j]="$ai"
        fi
    done
done

MAIN_GPU_ORDER_IDX=0
MAIN_GPU_ARRAY_IDX="${SORTED_ORDER[$MAIN_GPU_ORDER_IDX]}"
MAIN_GPU_ID="${GPU_IDS[$MAIN_GPU_ARRAY_IDX]}"

# ---------------------------------------------------------------------------
# Build --tensor-split string (usable MiB for each GPU in GPU-ID order)
# ---------------------------------------------------------------------------
TENSOR_SPLIT=""
for (( i=0; i<GPU_COUNT; i++ )); do
    TENSOR_SPLIT+="${USABLE_MIB[$i]}"
    if (( i < GPU_COUNT - 1 )); then TENSOR_SPLIT+=","; fi
done

# Total usable VRAM for logging
TOTAL_USABLE=0
for mib in "${USABLE_MIB[@]}"; do TOTAL_USABLE=$(( TOTAL_USABLE + mib )); done

# ---------------------------------------------------------------------------
# Print configuration summary
# ---------------------------------------------------------------------------
echo "========================================================"
echo "  TurboQuant llama-server — Multi-GPU Launch Config"
echo "========================================================"
echo ""
echo "  GPUs:"
for (( i=0; i<GPU_COUNT; i++ )); do
    total_raw_check=$(nvidia-smi --id="${GPU_IDS[$i]}" --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null || echo "?")
    marker=""; [[ "${GPU_IDS[$i]}" == "$MAIN_GPU_ID" ]] && marker=" ★ main"
    echo "    GPU ${GPU_IDS[$i]}: ${GPU_NAMES[$i]}"
    echo "           total VRAM  : ${total_raw_check} MiB"
    echo "           reserve (${VRAM_RESERVE_PCT}%): $(( total_raw_check * VRAM_RESERVE_PCT / 100 )) MiB${marker}"
    echo "           usable      : ${USABLE_MIB[$i]} MiB"
done
echo ""
echo "  Total usable VRAM  : ${TOTAL_USABLE} MiB ($(( TOTAL_USABLE / 1024 )) GiB)"
echo "  Main GPU           : GPU ${MAIN_GPU_ID} (${GPU_NAMES[$MAIN_GPU_ARRAY_IDX]})"
echo "  Tensor split       : ${TENSOR_SPLIT}"
echo "  KV cache type      : ${CACHE_TYPE} K + ${CACHE_TYPE} V"
echo "  Context size       : ${CTX_SIZE}"
echo "  Model              : ${MODEL_PATH}"
echo ""

# ---------------------------------------------------------------------------
# Build and run the command
# ---------------------------------------------------------------------------
CMD=(
    "$LLAMA_BIN"
    -m "$MODEL_PATH"
    --alias "model-turbo"
    --jinja
    -ngl "$NGL"
    --main-gpu "$MAIN_GPU_ID"
    --tensor-split "$TENSOR_SPLIT"
    -c "$CTX_SIZE"
    -fa on
    --cache-type-k "$CACHE_TYPE"
    --cache-type-v "$CACHE_TYPE"
    -np 1
    --metrics
    --host 0.0.0.0
    --port "$PORT"
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
)

echo "  Command:"
echo "    ${CMD[*]}"
echo ""
echo "========================================================"
echo ""

exec "${CMD[@]}"
