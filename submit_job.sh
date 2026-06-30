#!/bin/bash

#SBATCH --job-name=roof_pipeline
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=roof_pipeline_%j.out
#SBATCH --error=roof_pipeline_%j.err
#SBATCH --account=<CINECA_ACCOUNT>

set -euo pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

# Optional if needed on Leonardo
# module load singularity
# or
# module load apptainer

# Path to your Singularity/Apptainer image
SIF=${SIF:-${SLURM_SUBMIT_DIR}/cityzen-pipeline.sif}

# Project root on Leonardo host
HOST_CITYZEN=${HOST_CITYZEN:-${SLURM_SUBMIT_DIR}}

# Make sure output directory exists on host
mkdir -p "${HOST_CITYZEN}/data/output"

export DSM_BATCH_SIZE="${DSM_BATCH_SIZE:-16}"
export DSMNET_DATASET="${DSMNET_DATASET:-Bologna}"
export DSMNET_CHECKPOINT_DIR="${DSMNET_CHECKPOINT_DIR:-/workspace/DSMNet/checkpoints/Bologna}"
export DSMNET_NUM_CLASSES="${DSMNET_NUM_CLASSES:-2}"
export DSMNET_BUILDING_CLASS_INDEX="${DSMNET_BUILDING_CLASS_INDEX:-1}"
export NDSM_HEIGHT_SCALE="${NDSM_HEIGHT_SCALE:-1}"
export NDSM_HEIGHT_OFFSET="${NDSM_HEIGHT_OFFSET:-0}"
export NDSM_CLAMP_MIN="${NDSM_CLAMP_MIN:-0}"
export DSM_CLAMP_MIN="${DSM_CLAMP_MIN:-${NDSM_CLAMP_MIN}}"
# Leave the calibration directories empty to keep the default 1.0 runtime scale.
export NDSM_CALIBRATION_GT_DIR="${NDSM_CALIBRATION_GT_DIR:-}"
export NDSM_CALIBRATION_MASK_DIR="${NDSM_CALIBRATION_MASK_DIR:-}"
export NDSM_CALIBRATION_WLD_DIR="${NDSM_CALIBRATION_WLD_DIR:-}"
export NDSM_CALIBRATION_MODE="${NDSM_CALIBRATION_MODE:-scale}"
export NDSM_CALIBRATION_MIN_BUILDINGS="${NDSM_CALIBRATION_MIN_BUILDINGS:-10}"
export NDSM_CALIBRATION_MIN_COMPONENT_PIXELS="${NDSM_CALIBRATION_MIN_COMPONENT_PIXELS:-16}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export ABSL_MIN_LOG_LEVEL="${ABSL_MIN_LOG_LEVEL:-3}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

srun singularity exec --nv \
    --bind ${HOST_CITYZEN}:/workspace \
    ${SIF} \
    /bin/bash -lc "cd /workspace && \
        export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL} && \
        export ABSL_MIN_LOG_LEVEL=${ABSL_MIN_LOG_LEVEL} && \
        export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING} && \
        export CITYZEN_DSMNET_DATASET='${DSMNET_DATASET}' && \
        export CITYZEN_DSMNET_CHECKPOINT_DIR='${DSMNET_CHECKPOINT_DIR}' && \
        export CITYZEN_DSMNET_NUM_CLASSES='${DSMNET_NUM_CLASSES}' && \
        export CITYZEN_DSMNET_BUILDING_CLASS_INDEX=${DSMNET_BUILDING_CLASS_INDEX} && \
        export CITYZEN_DSM_CLAMP_MIN=${DSM_CLAMP_MIN} && \
        export CITYZEN_NDSM_HEIGHT_SCALE=${NDSM_HEIGHT_SCALE} && \
        export CITYZEN_NDSM_HEIGHT_OFFSET=${NDSM_HEIGHT_OFFSET} && \
        export CITYZEN_NDSM_CLAMP_MIN=${NDSM_CLAMP_MIN} && \
        export CITYZEN_NDSM_CALIBRATION_GT_DIR='${NDSM_CALIBRATION_GT_DIR}' && \
        export CITYZEN_NDSM_CALIBRATION_MASK_DIR='${NDSM_CALIBRATION_MASK_DIR}' && \
        export CITYZEN_NDSM_CALIBRATION_WLD_DIR='${NDSM_CALIBRATION_WLD_DIR}' && \
        export CITYZEN_NDSM_CALIBRATION_MODE=${NDSM_CALIBRATION_MODE} && \
        export CITYZEN_NDSM_CALIBRATION_MIN_BUILDINGS=${NDSM_CALIBRATION_MIN_BUILDINGS} && \
        export CITYZEN_NDSM_CALIBRATION_MIN_COMPONENT_PIXELS=${NDSM_CALIBRATION_MIN_COMPONENT_PIXELS} && \
        python3.11 /workspace/pipeline.py \
        --ortho_path /workspace/data/input/ortho/ \
        --footprints_path /workspace/data/input/footprints/ \
        --output_dir /workspace/data/output/ \
        --dsm_batch_size ${DSM_BATCH_SIZE} \
        --ndsm_height_scale ${NDSM_HEIGHT_SCALE} \
        --ndsm_height_offset ${NDSM_HEIGHT_OFFSET} \
        --ndsm_clamp_min ${NDSM_CLAMP_MIN} \
        --ndsm_calibration_mode ${NDSM_CALIBRATION_MODE} \
        --ndsm_calibration_min_buildings ${NDSM_CALIBRATION_MIN_BUILDINGS} \
        --ndsm_calibration_min_component_pixels ${NDSM_CALIBRATION_MIN_COMPONENT_PIXELS}" \
    2> >(awk '
        /WARNING: All log messages before absl::InitializeLog\(\) is called are written to STDERR/ { next }
        /gpu_timer\.cc:114\] Skipping the delay kernel, measurement accuracy will be reduced/ { next }
        { print > "/dev/stderr" }
    ')

echo "End time: $(date)"
