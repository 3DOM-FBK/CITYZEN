import logging
import os

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from shapely.geometry import mapping

from utils import footprints_base as base_footprints


classify_footprint_image = base_footprints.classify_footprint_image
pixel_to_geographic = base_footprints.pixel_to_geographic
print_footprint_summary = base_footprints.print_footprint_summary


def _prepare_geometry(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    return geometry


def _geometry_window_from_bounds(src, geometry, buffer_pixels=0):
    minx, miny, maxx, maxy = geometry.bounds
    transform = src.transform
    px1, py1 = ~transform * (minx, maxy)
    px2, py2 = ~transform * (maxx, miny)

    row_start = max(0, int(min(py1, py2)) - buffer_pixels)
    row_end = min(src.height, int(max(py1, py2)) + buffer_pixels + 1)
    col_start = max(0, int(min(px1, px2)) - buffer_pixels)
    col_end = min(src.width, int(max(px1, px2)) + buffer_pixels + 1)

    if row_end <= row_start or col_end <= col_start:
        return None
    return row_start, row_end, col_start, col_end


def _geometry_mask_for_window(src, geometry, row_start, row_end, col_start, col_end):
    window = Window.from_slices((row_start, row_end), (col_start, col_end))
    return geometry_mask(
        [mapping(geometry)],
        transform=src.window_transform(window),
        invert=True,
        out_shape=(row_end - row_start, col_end - col_start),
        all_touched=True,
    )


def extract_footprint_height(dsm_path, footprint_geometry, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    try:
        with rasterio.open(dsm_path) as dsm_src:
            footprint_geometry = _prepare_geometry(footprint_geometry)
            if footprint_geometry is None:
                return None

            if dsm_src.crs is None or dsm_src.transform is None:
                logger.error("DSM is not georeferenced - cannot extract height for footprint")
                return None

            window_bounds = _geometry_window_from_bounds(dsm_src, footprint_geometry, buffer_pixels=0)
            if window_bounds is None:
                return None

            row_start, row_end, col_start, col_end = window_bounds
            dsm_window = dsm_src.read(1, window=((row_start, row_end), (col_start, col_end)))
            footprint_mask = _geometry_mask_for_window(
                dsm_src,
                footprint_geometry,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            if not np.any(footprint_mask):
                return None

            if dsm_src.nodata is not None:
                valid_mask = (dsm_window != dsm_src.nodata) & footprint_mask
            else:
                if dsm_src.dtypes[0] == "uint8":
                    valid_mask = np.ones_like(dsm_window, dtype=bool) & footprint_mask
                else:
                    valid_mask = (
                        (dsm_window != -9999)
                        & (dsm_window != -32768)
                        & (~np.isnan(dsm_window))
                        & footprint_mask
                    )

            if not np.any(valid_mask):
                return None

            dsm_values = dsm_window[valid_mask]
            if len(dsm_values) > 0:
                return {
                    "mean_height": float(np.mean(dsm_values)),
                    "min_height": float(np.min(dsm_values)),
                    "max_height": float(np.max(dsm_values)),
                    "std_height": float(np.std(dsm_values)),
                    "pixel_count": len(dsm_values),
                }
            return None
    except Exception as e:
        logger.error(f"Error extracting height: {e}")
        return None


def extract_footprint_image(orthophoto_path, footprint_geometry, buffer_pixels=10, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    try:
        with rasterio.open(orthophoto_path) as src:
            footprint_geometry = _prepare_geometry(footprint_geometry)
            if footprint_geometry is None:
                logger.error("Invalid footprint geometry")
                return None

            window_bounds = _geometry_window_from_bounds(src, footprint_geometry, buffer_pixels=buffer_pixels)
            if window_bounds is None:
                logger.error("Invalid footprint bounds")
                return None

            row_start, row_end, col_start, col_end = window_bounds
            image_data = src.read(window=((row_start, row_end), (col_start, col_end)))
            footprint_mask = _geometry_mask_for_window(
                src,
                footprint_geometry,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            if not np.any(footprint_mask):
                logger.error("Footprint mask is empty")
                return None

            if image_data.shape[0] == 1:
                image_array = np.stack([image_data[0]] * 3, axis=0)
            elif image_data.shape[0] >= 3:
                image_array = image_data[:3]
            else:
                logger.error(f"Unsupported number of channels: {image_data.shape[0]}")
                return None

            image_array = np.transpose(image_array, (1, 2, 0))
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)

            image_array[~footprint_mask] = 0
            return base_footprints.Image.fromarray(image_array)
    except Exception as e:
        logger.error(f"Error extracting footprint from {os.path.basename(orthophoto_path)}: {e}")
        return None


def process_footprints(*args, **kwargs):
    original_extract_height = base_footprints.extract_footprint_height
    original_extract_image = base_footprints.extract_footprint_image
    try:
        base_footprints.extract_footprint_height = extract_footprint_height
        base_footprints.extract_footprint_image = extract_footprint_image
        return base_footprints.process_footprints(*args, **kwargs)
    finally:
        base_footprints.extract_footprint_height = original_extract_height
        base_footprints.extract_footprint_image = original_extract_image
