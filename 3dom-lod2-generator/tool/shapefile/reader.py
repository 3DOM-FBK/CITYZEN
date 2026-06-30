import logging
import os
import sys
from collections import Counter

import geopandas as gpd
import pandas as pd


project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(project_root, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from io_utils.debug import print_to_terminal


log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info(f"Shapefile Reader: Initialisation avec niveau de log: {log_level}")


MIN_REASONABLE_HEIGHT = float(os.environ.get("CITYZEN_MIN_ABSOLUTE_HEIGHT", "2.0"))
MAX_REASONABLE_HEIGHT = float(os.environ.get("CITYZEN_MAX_ABSOLUTE_HEIGHT", "200.0"))

ROOF_TYPE_MAPPING = {
    "flat": "flat",
    "gable": "gabled",
    "gabled": "gabled",
    "l-shaped": "gabled-L",
    "gabled-l": "gabled-L",
    "gabled_l": "gabled-L",
    "hip": "hip",
    "hipped": "hip",
    "pyramid": "pyramid",
    "pyramidal": "pyramid",
    "complex": "complex",
    "unknown": "unknown",
}

CLASSIFIED_ROOF_COLUMNS = ("roof_class", "roof_cls", "roof_type")
FALLBACK_ROOF_COLUMNS = ("roof",)
HEIGHT_COLUMNS = ("max_height", "max_hgt", "height", "mean_height", "mean_hgt")
PROBABILITY_COLUMNS = {
    "complex": ("prob_complex", "prob_cmplx"),
    "flat": ("prob_flat",),
    "gable": ("prob_gable",),
    "hip": ("prob_hip",),
    "l-shaped": ("prob_L-shaped", "prob_lshp"),
    "pyramid": ("prob_pyramid", "prob_pyrmd"),
}


def _normalize_roof_value(value):
    if value is None or pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized in {"nan", "null", "none"}:
        return None
    return normalized


def _coerce_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _row_value(row, columns):
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        return value
    return None


def _resolve_roof_class(row):
    classified_roof = _normalize_roof_value(_row_value(row, CLASSIFIED_ROOF_COLUMNS))
    fallback_roof = _normalize_roof_value(_row_value(row, FALLBACK_ROOF_COLUMNS))

    # Prefer the classifier output, but keep the original footprint roof value as a
    # fallback when the classifier produced "unknown" or "complex".
    if classified_roof in {None, "unknown"} and fallback_roof:
        return fallback_roof
    if classified_roof == "complex" and fallback_roof not in {None, "complex", "unknown"}:
        return fallback_roof
    return classified_roof or fallback_roof or "unknown"


def _resolve_height(row):
    for column in HEIGHT_COLUMNS:
        if column not in row.index:
            continue
        value = _coerce_float(row.get(column))
        if value is None:
            continue

        # CITYZEN DSM outputs are currently normalized (~0-1), so tiny values
        # should not override the point-cloud-derived building height.
        if value < MIN_REASONABLE_HEIGHT or value > MAX_REASONABLE_HEIGHT:
            continue
        return value
    return None


def _resolve_roof_probabilities(row):
    probabilities = {}
    for roof_label, columns in PROBABILITY_COLUMNS.items():
        value = _coerce_float(_row_value(row, columns))
        if value is None:
            continue
        probabilities[roof_label] = max(0.0, min(1.0, value))
    return probabilities


def read_shapefile_polygons(shapefile_path):
    """
    Read a vector file and return:
    - a list of dicts with { 'exterior': [...], 'holes': [...], 'roof': ..., 'height': ... }
    - coordinate offset (x_offset, y_offset)

    GeoJSON is preferred in the pipeline because it preserves long field names,
    but any vector file supported by GeoPandas can be read here.
    """
    logger.info(f"Reading vector file: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    logger.info(f"Loaded {len(gdf)} features from vector file")
    logger.debug(f"Available columns: {list(gdf.columns)}")

    polygons = []

    total_bounds = gdf.total_bounds
    x_offset, y_offset = total_bounds[0], total_bounds[1]
    logger.debug(f"Coordinate offset: x_offset={x_offset}, y_offset={y_offset}")

    resolved_roof_counts = Counter()
    discarded_height_count = 0

    def process_coords(coords):
        return [
            (x - x_offset, y - y_offset, z if len(coord) == 3 else 0)
            for coord in coords
            for x, y, *z_list in [coord]
            for z in [(z_list[0] if z_list else 0)]
        ]

    for idx, row in gdf.iterrows():
        geom = row.geometry

        raw_roof = _resolve_roof_class(row)
        mapped_roof_type = ROOF_TYPE_MAPPING.get(raw_roof, raw_roof or "unknown")
        roof_probabilities = _resolve_roof_probabilities(row)

        height = _resolve_height(row)
        if height is None:
            for column in HEIGHT_COLUMNS:
                if column in row.index and _coerce_float(row.get(column)) is not None:
                    discarded_height_count += 1
                    break

        resolved_roof_counts[mapped_roof_type] += 1

        if idx < 5:
            logger.debug(
                f"Building {idx}: roof_class='{raw_roof}' -> mapped='{mapped_roof_type}', "
                f"height={height}"
            )

        if geom.geom_type == "Polygon":
            exterior = process_coords(geom.exterior.coords)
            holes = [process_coords(interior.coords) for interior in geom.interiors]
            polygons.append(
                {
                    "exterior": exterior,
                    "holes": holes,
                    "roof": mapped_roof_type,
                    "roof_source": raw_roof,
                    "roof_probabilities": roof_probabilities,
                    "height": height,
                }
            )
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                exterior = process_coords(poly.exterior.coords)
                holes = [process_coords(interior.coords) for interior in poly.interiors]
                polygons.append(
                    {
                        "exterior": exterior,
                        "holes": holes,
                        "roof": mapped_roof_type,
                        "roof_source": raw_roof,
                        "roof_probabilities": roof_probabilities,
                        "height": height,
                    }
                )

    logger.info(f"Created {len(polygons)} polygon records")
    logger.info(f"Resolved roof distribution for 3DOM: {dict(resolved_roof_counts)}")
    if discarded_height_count:
        logger.info(
            "Ignored %d small or implausible stored height values and will use DSM-derived "
            "heights instead",
            discarded_height_count,
        )

    return polygons, (x_offset, y_offset)
