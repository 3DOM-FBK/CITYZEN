#!/bin/bash

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_DIR}"

if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile ]; then
        # Load the site shell initialization so the "module" function is available.
        source /etc/profile
    fi
fi

if command -v module >/dev/null 2>&1; then
    module purge
    module load hpcx-mpi/2.19
    module load cuda/12.2
fi

mkdir -p "${REPO_DIR}/.singularity-cache" "${REPO_DIR}/.singularity-tmp" "${REPO_DIR}/data/output"
export SINGULARITY_CACHEDIR="${REPO_DIR}/.singularity-cache"
export SINGULARITY_TMPDIR="${REPO_DIR}/.singularity-tmp"

if [ ! -f "${REPO_DIR}/cityzen-pipeline.tar" ]; then
    echo "Missing Docker archive: ${REPO_DIR}/cityzen-pipeline.tar"
    exit 1
fi

echo "Building SIF from ${REPO_DIR}/cityzen-pipeline.tar"
singularity build "${REPO_DIR}/cityzen-pipeline.sif" "docker-archive://${REPO_DIR}/cityzen-pipeline.tar"

echo "SIF ready at ${REPO_DIR}/cityzen-pipeline.sif"
