#!/bin/bash
set -euo pipefail

ORTHO_PATH="/workspace/data/input/ortho"
FOOTPRINTS_PATH=""
OUTPUT_DIR="/workspace/data/output"
DSM_BATCH_SIZE="${DSM_BATCH_SIZE:-16}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ortho_path)
            ORTHO_PATH="$2"
            shift 2
            ;;
        --footprints_path)
            FOOTPRINTS_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            python -u /workspace/pipeline.py --help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            python -u /workspace/pipeline.py --help
            exit 1
            ;;
    esac
done

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export ABSL_MIN_LOG_LEVEL="${ABSL_MIN_LOG_LEVEL:-3}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

if [[ -z "${FOOTPRINTS_PATH}" ]]; then
    echo "Error: --footprints_path is required and must point to a shapefile or directory of shapefiles." >&2
    python -u /workspace/pipeline.py --help
    exit 1
fi

cmd=(
    python -u /workspace/pipeline.py
    --ortho_path "$ORTHO_PATH"
    --footprints_path "$FOOTPRINTS_PATH"
    --output_dir "$OUTPUT_DIR"
    --dsm_batch_size "$DSM_BATCH_SIZE"
)

exec "${cmd[@]}"
