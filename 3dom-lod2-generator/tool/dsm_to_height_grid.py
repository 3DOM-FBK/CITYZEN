#!/usr/bin/env python3
"""
Convert a DSM / nDSM raster into a calibrated height grid for Blender.

This keeps GeoTIFF reading outside Blender's embedded Python so
reconstruction can still use raster-based height queries even when Blender
does not have rasterio installed.
"""

import argparse
import logging
import os
import sys

import numpy as np
import rasterio


log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info(f"DSM to Height Grid: Initialisation avec niveau de log: {log_level}")


def dsm_to_height_grid(dsm_path, output_path, z_scale=1.0, z_offset=0.0, clamp_min=0.0):
    logger.info("Converting DSM %s to calibrated height grid", dsm_path)
    logger.info("Applying vertical calibration: z = dsm * %.3f + %.3f", z_scale, z_offset)
    logger.info("Applying minimum-height clamp: %.3f", clamp_min)

    with rasterio.open(dsm_path) as src:
        heights = src.read(1).astype(np.float32, copy=False)
        transform = src.transform

        if src.nodata is not None:
            valid_mask = heights != src.nodata
        else:
            valid_mask = np.isfinite(heights)

        calibrated = np.full(heights.shape, np.nan, dtype=np.float32)
        calibrated_values = heights[valid_mask] * np.float32(z_scale) + np.float32(z_offset)
        clipped_low_count = int(np.count_nonzero(calibrated_values < np.float32(clamp_min)))
        if clipped_low_count:
            calibrated_values = np.maximum(calibrated_values, np.float32(clamp_min))
        calibrated[valid_mask] = calibrated_values

        transform_values = np.array(
            [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f],
            dtype=np.float64,
        )
        inverse_values = np.array(
            [(~transform).a, (~transform).b, (~transform).c, (~transform).d, (~transform).e, (~transform).f],
            dtype=np.float64,
        )

    np.savez_compressed(
        output_path,
        heights=calibrated,
        transform=transform_values,
        inverse_transform=inverse_values,
    )
    logger.info("Height grid saved to: %s", output_path)
    logger.info("Grid shape: %s", calibrated.shape)
    if clipped_low_count:
        logger.info("Clamped %d pixels below %.3f", clipped_low_count, float(clamp_min))
    finite_values = calibrated[np.isfinite(calibrated)]
    if finite_values.size:
        logger.info(
            "Calibrated height range: %.3f to %.3f",
            float(np.min(finite_values)),
            float(np.max(finite_values)),
        )


def main():
    parser = argparse.ArgumentParser(description="Convert DSM to calibrated height grid for Blender")
    parser.add_argument("-i", "--input", required=True, help="Input DSM TIF file")
    parser.add_argument("-o", "--output", required=True, help="Output NPZ file")
    parser.add_argument(
        "--z_scale",
        type=float,
        default=float(os.environ.get("CITYZEN_NDSM_HEIGHT_SCALE", "1.0")),
        help="Multiply normalized nDSM values by this factor before export",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=float(os.environ.get("CITYZEN_NDSM_HEIGHT_OFFSET", "0.0")),
        help="Add a vertical offset after scaling",
    )
    parser.add_argument(
        "--clamp_min",
        type=float,
        default=float(os.environ.get("CITYZEN_NDSM_CLAMP_MIN", "0.0")),
        help="Clamp calibrated heights below this value to avoid underground outliers",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error("Error: Input file not found: %s", args.input)
        return 1

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    try:
        dsm_to_height_grid(
            args.input,
            args.output,
            args.z_scale,
            args.z_offset,
            args.clamp_min,
        )
        return 0
    except Exception as exc:
        logger.error("Error converting DSM to height grid: %s", exc)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
