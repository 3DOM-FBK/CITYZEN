#!/usr/bin/env python3
"""
Fit a vertical calibration for CITYZEN nDSM rasters from georeferenced
reference patches.

This is intentionally generic: it only assumes
- a predicted CITYZEN raster,
- a directory of reference nDSM patches,
- a directory of world files describing patch placement,
- and optionally a directory of binary building-mask patches.

The default calibration uses per-building median heights and a scale-only fit
so zero-height pixels remain zero after calibration.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _read_world_file(world_path: Path):
    values = [float(line.strip()) for line in world_path.read_text().splitlines()[:6]]
    if len(values) != 6:
        raise ValueError(f"World file {world_path} does not contain 6 lines")
    pixel_x, _, _, pixel_y, center_x, center_y = values
    left = center_x - (pixel_x / 2.0)
    top = center_y - (pixel_y / 2.0)
    return {
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "left": left,
        "top": top,
    }


def _strip_known_suffix(stem: str, ndsm_suffix: str):
    ndsm_stem = Path(ndsm_suffix).stem
    if ndsm_stem and stem.endswith(ndsm_stem):
        return stem[: -len(ndsm_stem)]
    return stem


def _derive_patch_id(world_path: Path, ndsm_suffix: str):
    return _strip_known_suffix(world_path.stem, ndsm_suffix)


def _safe_read_raster(path: Path):
    with rasterio.open(path) as src:
        return src.read(1)


def _component_medians(pred_patch, gt_patch, mask_patch, min_component_pixels):
    mask_patch = mask_patch.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_patch, connectivity=4)
    pred_vals = []
    gt_vals = []
    for label in range(1, num_labels):
        selector = labels == label
        if int(selector.sum()) < min_component_pixels:
            continue
        pred_vals.append(float(np.median(pred_patch[selector])))
        gt_vals.append(float(np.median(gt_patch[selector])))
    return pred_vals, gt_vals


def _fit_scale_only(x, y):
    denom = float(np.dot(x, x))
    if denom <= 0:
        raise ValueError("Cannot fit scale-only calibration with zero prediction variance")
    return float(np.dot(x, y) / denom), 0.0


def _fit_affine(x, y):
    design = np.column_stack([x, np.ones_like(x)])
    scale, offset = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(scale), float(offset)


def _pair_stats(pred, gt):
    diff = pred - gt
    return {
        "count": int(diff.size),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(math.sqrt(np.mean(diff * diff))),
        "bias": float(np.mean(diff)),
    }


def iter_overlapping_reference_patches(
    pred_raster_path,
    reference_ndsm_dir,
    reference_world_dir,
    reference_mask_dir=None,
    ndsm_suffix="_AGL.tif",
    mask_suffix="_BLG.tif",
    world_suffix=".tfw",
    require_full_inside=True,
):
    pred_raster_path = Path(pred_raster_path)
    reference_ndsm_dir = Path(reference_ndsm_dir)
    reference_world_dir = Path(reference_world_dir)
    reference_mask_dir = Path(reference_mask_dir) if reference_mask_dir else None

    with rasterio.open(pred_raster_path) as pred_src:
        pred_left = pred_src.bounds.left
        pred_top = pred_src.bounds.top
        pixel_x = pred_src.transform.a
        pixel_y = pred_src.transform.e

        for world_path in sorted(reference_world_dir.glob(f"*{world_suffix}")):
            patch_id = _derive_patch_id(world_path, ndsm_suffix)
            gt_path = reference_ndsm_dir / f"{patch_id}{ndsm_suffix}"
            if not gt_path.exists():
                continue

            mask_path = None
            if reference_mask_dir:
                candidate_mask = reference_mask_dir / f"{patch_id}{mask_suffix}"
                if candidate_mask.exists():
                    mask_path = candidate_mask

            world = _read_world_file(world_path)
            with rasterio.open(gt_path) as gt_src:
                patch_width = gt_src.width
                patch_height = gt_src.height

            col = int(round((world["left"] - pred_left) / pixel_x))
            row = int(round((pred_top - world["top"]) / abs(pixel_y)))

            inside = (
                row >= 0
                and col >= 0
                and row + patch_height <= pred_src.height
                and col + patch_width <= pred_src.width
            )
            if require_full_inside and not inside:
                continue
            if not inside:
                continue

            pred_patch = pred_src.read(1, window=Window(col, row, patch_width, patch_height))
            gt_patch = _safe_read_raster(gt_path)
            if mask_path is not None:
                mask_patch = _safe_read_raster(mask_path) > 0
            else:
                mask_patch = gt_patch > 0

            yield {
                "patch_id": patch_id,
                "pred_patch": pred_patch.astype(np.float64, copy=False),
                "gt_patch": gt_patch.astype(np.float64, copy=False),
                "mask_patch": mask_patch.astype(bool, copy=False),
            }


def fit_height_calibration(
    pred_raster_path,
    reference_ndsm_dir,
    reference_world_dir,
    reference_mask_dir=None,
    mode="scale",
    min_buildings=10,
    min_component_pixels=16,
    ndsm_suffix="_AGL.tif",
    mask_suffix="_BLG.tif",
    world_suffix=".tfw",
):
    supported_modes = {"scale", "affine"}
    if mode not in supported_modes:
        raise ValueError(f"Unsupported calibration mode: {mode}")

    pred_samples = []
    gt_samples = []
    patch_ids = []

    for patch in iter_overlapping_reference_patches(
        pred_raster_path=pred_raster_path,
        reference_ndsm_dir=reference_ndsm_dir,
        reference_world_dir=reference_world_dir,
        reference_mask_dir=reference_mask_dir,
        ndsm_suffix=ndsm_suffix,
        mask_suffix=mask_suffix,
        world_suffix=world_suffix,
        require_full_inside=True,
    ):
        patch_ids.append(patch["patch_id"])
        pred_vals, gt_vals = _component_medians(
            patch["pred_patch"],
            patch["gt_patch"],
            patch["mask_patch"],
            min_component_pixels=min_component_pixels,
        )
        pred_samples.extend(pred_vals)
        gt_samples.extend(gt_vals)

    if len(pred_samples) < min_buildings:
        raise ValueError(
            f"Only {len(pred_samples)} building samples available for calibration; "
            f"need at least {min_buildings}"
        )

    x = np.asarray(pred_samples, dtype=np.float64)
    y = np.asarray(gt_samples, dtype=np.float64)

    if mode == "scale":
        scale, offset = _fit_scale_only(x, y)
    else:
        scale, offset = _fit_affine(x, y)

    calibrated = x * scale + offset

    return {
        "status": "ok",
        "pred_raster_path": str(pred_raster_path),
        "reference_ndsm_dir": str(reference_ndsm_dir),
        "reference_world_dir": str(reference_world_dir),
        "reference_mask_dir": str(reference_mask_dir) if reference_mask_dir else None,
        "mode": mode,
        "scale": float(scale),
        "offset": float(offset),
        "patch_count": len(set(patch_ids)),
        "building_count": int(x.size),
        "sample_pred_min": float(x.min()),
        "sample_pred_max": float(x.max()),
        "sample_gt_min": float(y.min()),
        "sample_gt_max": float(y.max()),
        "raw_per_building": _pair_stats(x, y),
        "calibrated_per_building": _pair_stats(calibrated, y),
        "patch_id_examples": sorted(set(patch_ids))[:20],
        "ndsm_suffix": ndsm_suffix,
        "mask_suffix": mask_suffix,
        "world_suffix": world_suffix,
        "min_buildings": int(min_buildings),
        "min_component_pixels": int(min_component_pixels),
    }


def main():
    parser = argparse.ArgumentParser(description="Fit a vertical nDSM calibration from reference patches")
    parser.add_argument("--pred_raster", required=True, help="CITYZEN nDSM / DSM raster")
    parser.add_argument("--reference_ndsm_dir", required=True, help="Reference nDSM patch directory")
    parser.add_argument("--reference_world_dir", required=True, help="Directory containing world files")
    parser.add_argument("--reference_mask_dir", help="Optional reference building-mask patch directory")
    parser.add_argument("--mode", choices=["scale", "affine"], default="scale", help="Calibration model")
    parser.add_argument("--min_buildings", type=int, default=10, help="Minimum building samples required")
    parser.add_argument("--min_component_pixels", type=int, default=16, help="Ignore tiny components below this size")
    parser.add_argument("--ndsm_suffix", default="_AGL.tif", help="Reference nDSM filename suffix")
    parser.add_argument("--mask_suffix", default="_BLG.tif", help="Reference mask filename suffix")
    parser.add_argument("--world_suffix", default=".tfw", help="World file suffix")
    parser.add_argument("--output_json", help="Optional JSON report path")
    args = parser.parse_args()

    report = fit_height_calibration(
        pred_raster_path=args.pred_raster,
        reference_ndsm_dir=args.reference_ndsm_dir,
        reference_world_dir=args.reference_world_dir,
        reference_mask_dir=args.reference_mask_dir,
        mode=args.mode,
        min_buildings=args.min_buildings,
        min_component_pixels=args.min_component_pixels,
        ndsm_suffix=args.ndsm_suffix,
        mask_suffix=args.mask_suffix,
        world_suffix=args.world_suffix,
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
