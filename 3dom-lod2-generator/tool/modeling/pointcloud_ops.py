import logging
import os
import sys

import numpy as np

log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info(f"Pointcloud Ops: Initialisation avec niveau de log: {log_level}")


MIN_HEIGHT_DELTA = float(os.environ.get("CITYZEN_MIN_BUILDING_HEIGHT_DELTA", "2.0"))
DEFAULT_HEIGHT_RANGE = (
    float(os.environ.get("CITYZEN_DEFAULT_Z_MIN", "0.0")),
    float(os.environ.get("CITYZEN_DEFAULT_Z_MAX", "10.0")),
)


def _normalize_min_max(z_min, z_max):
    if z_min is None or z_max is None:
        return DEFAULT_HEIGHT_RANGE

    z_min = float(z_min)
    z_max = float(z_max)
    if (z_max - z_min) < MIN_HEIGHT_DELTA:
        z_max = z_min + MIN_HEIGHT_DELTA
    return z_min, z_max


def _summarize_z_values(z_values):
    if z_values is None or getattr(z_values, "size", 0) == 0:
        return {
            "count": 0,
            "z_min": None,
            "z_max": None,
            "relief": None,
            "z_mean": None,
            "z_std": None,
        }

    z_values = np.asarray(z_values, dtype=np.float32)
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    return {
        "count": int(z_values.size),
        "z_min": z_min,
        "z_max": z_max,
        "relief": float(z_max - z_min),
        "z_mean": float(np.mean(z_values)),
        "z_std": float(np.std(z_values)),
    }


def _polygon_shell_and_holes(poly, x_offset, y_offset):
    exterior = [
        (coord[0] + x_offset, coord[1] + y_offset)
        for coord in poly.get("exterior", [])
        if len(coord) >= 2
    ]
    holes = [
        [(coord[0] + x_offset, coord[1] + y_offset) for coord in hole if len(coord) >= 2]
        for hole in poly.get("holes", [])
    ]
    holes = [hole for hole in holes if len(hole) >= 3]
    return exterior, holes


def _ring_as_array(ring):
    ring_array = np.asarray(ring, dtype=np.float64)
    if ring_array.shape[0] == 0:
        return ring_array
    if not np.allclose(ring_array[0], ring_array[-1]):
        ring_array = np.vstack([ring_array, ring_array[0]])
    return ring_array


def _points_in_ring(x_coords, y_coords, ring):
    ring_array = _ring_as_array(ring)
    if ring_array.shape[0] < 4:
        return np.zeros_like(x_coords, dtype=bool)

    x1 = ring_array[:-1, 0]
    y1 = ring_array[:-1, 1]
    x2 = ring_array[1:, 0]
    y2 = ring_array[1:, 1]

    inside = np.zeros_like(x_coords, dtype=bool)
    for x_start, y_start, x_end, y_end in zip(x1, y1, x2, y2):
        if abs(y_end - y_start) < 1e-12:
            continue
        crosses = ((y_start > y_coords) != (y_end > y_coords))
        x_intersection = ((x_end - x_start) * (y_coords - y_start) / (y_end - y_start)) + x_start
        inside ^= crosses & (x_coords < x_intersection)
    return inside


def _points_in_polygon(x_coords, y_coords, exterior, holes):
    inside = _points_in_ring(x_coords, y_coords, exterior)
    if not np.any(inside):
        return inside

    for hole in holes:
        inside &= ~_points_in_ring(x_coords, y_coords, hole)
    return inside


def _invert_affine(transform):
    a, b, c, d, e, f = [float(value) for value in transform]
    determinant = a * e - b * d
    if abs(determinant) < 1e-12:
        raise ValueError("DSM transform is not invertible")

    return (
        e / determinant,
        -b / determinant,
        (b * f - c * e) / determinant,
        -d / determinant,
        a / determinant,
        (c * d - a * f) / determinant,
    )


def _apply_affine(transform, cols, rows):
    a, b, c, d, e, f = transform
    x_coords = a * cols + b * rows + c
    y_coords = d * cols + e * rows + f
    return x_coords, y_coords


def _apply_inverse_affine(inverse_transform, x_coords, y_coords):
    return _apply_affine(inverse_transform, x_coords, y_coords)


class GridHeightSampler:
    def __init__(self, grid_path):
        payload = np.load(grid_path)
        self.heights = payload["heights"].astype(np.float32, copy=False)
        self.transform = tuple(float(value) for value in payload["transform"].tolist())
        if "inverse_transform" in payload:
            self.inverse_transform = tuple(float(value) for value in payload["inverse_transform"].tolist())
        else:
            self.inverse_transform = _invert_affine(self.transform)

        self.height = int(self.heights.shape[0])
        self.width = int(self.heights.shape[1])
        logger.info(
            "GridHeightSampler: %s opened with shape=%s",
            grid_path,
            getattr(self.heights, "shape", None),
        )

    def close(self):
        return None

    def _window_from_polygon(self, exterior, holes):
        all_rings = [exterior] + list(holes)
        xs = np.asarray([point[0] for ring in all_rings for point in ring], dtype=np.float64)
        ys = np.asarray([point[1] for ring in all_rings for point in ring], dtype=np.float64)
        if xs.size == 0 or ys.size == 0:
            return None

        cols, rows = _apply_inverse_affine(self.inverse_transform, xs, ys)
        col_start = max(0, int(np.floor(np.min(cols))) - 1)
        col_stop = min(self.width, int(np.ceil(np.max(cols))) + 2)
        row_start = max(0, int(np.floor(np.min(rows))) - 1)
        row_stop = min(self.height, int(np.ceil(np.max(rows))) + 2)

        if row_start >= row_stop or col_start >= col_stop:
            return None
        return row_start, row_stop, col_start, col_stop

    def _sample_polygon_heights(self, poly, x_offset, y_offset, idx=None):
        exterior, holes = _polygon_shell_and_holes(poly, x_offset, y_offset)
        if len(exterior) < 3:
            logger.debug("Building %s: invalid polygon geometry, using default height", idx)
            return None

        window = self._window_from_polygon(exterior, holes)
        if window is None:
            logger.debug("Building %s: polygon outside DSM extent, using default height", idx)
            return None

        row_start, row_stop, col_start, col_stop = window
        data = self.heights[row_start:row_stop, col_start:col_stop]
        if data.size == 0:
            logger.debug("Building %s: empty DSM window, using default height", idx)
            return None

        cols = np.arange(col_start, col_stop, dtype=np.float64) + 0.5
        rows = np.arange(row_start, row_stop, dtype=np.float64) + 0.5
        col_grid, row_grid = np.meshgrid(cols, rows, indexing="xy")
        x_grid, y_grid = _apply_affine(self.transform, col_grid, row_grid)

        inside_polygon = _points_in_polygon(x_grid, y_grid, exterior, holes)
        valid_pixels = inside_polygon & np.isfinite(data)
        if not np.any(valid_pixels):
            logger.debug("Building %s: no valid DSM pixels in footprint, using default height", idx)
            return None

        z_values = data[valid_pixels].astype(np.float32, copy=False)
        logger.debug("Building %s: grid sampler used %d pixels", idx, int(z_values.size))
        return z_values

    def get_min_max(self, poly, x_offset, y_offset, idx=None):
        stats = self.get_stats(poly, x_offset, y_offset, idx=idx)
        if stats["z_min"] is None or stats["z_max"] is None:
            return DEFAULT_HEIGHT_RANGE
        return _normalize_min_max(stats["z_min"], stats["z_max"])

    def get_stats(self, poly, x_offset, y_offset, idx=None):
        return _summarize_z_values(self._sample_polygon_heights(poly, x_offset, y_offset, idx=idx))


class RasterHeightSampler:
    def __init__(self, dsm_raster_path, z_scale=1.0, z_offset=0.0, all_touched=True):
        try:
            import rasterio
            from rasterio.features import geometry_mask
            from rasterio.windows import Window
            from shapely.geometry import Polygon, mapping
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "RasterHeightSampler requires rasterio and shapely; use a precomputed "
                "height grid in Blender when those packages are unavailable."
            ) from exc

        self.rasterio = rasterio
        self.geometry_mask = geometry_mask
        self.Window = Window
        self.Polygon = Polygon
        self.mapping = mapping
        self.dsm_raster_path = dsm_raster_path
        self.z_scale = float(z_scale)
        self.z_offset = float(z_offset)
        self.all_touched = all_touched
        self.src = rasterio.open(dsm_raster_path)
        logger.info(
            "RasterHeightSampler: %s opened with z = raster * %.3f + %.3f",
            dsm_raster_path,
            self.z_scale,
            self.z_offset,
        )

    def close(self):
        if self.src:
            self.src.close()

    def _window_from_bounds(self, bounds):
        minx, miny, maxx, maxy = bounds
        try:
            row_a, col_a = self.src.index(minx, maxy)
            row_b, col_b = self.src.index(maxx, miny)
        except Exception:
            return None

        row_start = max(0, min(row_a, row_b) - 1)
        row_stop = min(self.src.height, max(row_a, row_b) + 2)
        col_start = max(0, min(col_a, col_b) - 1)
        col_stop = min(self.src.width, max(col_a, col_b) + 2)

        if row_start >= row_stop or col_start >= col_stop:
            return None

        return self.Window.from_slices((row_start, row_stop), (col_start, col_stop))

    def _sample_polygon_heights(self, poly, x_offset, y_offset, idx=None):
        exterior, holes = _polygon_shell_and_holes(poly, x_offset, y_offset)
        if len(exterior) < 3:
            logger.debug("Building %s: invalid polygon geometry, using default height", idx)
            return None

        polygon = self.Polygon(exterior, holes=holes)
        if polygon.is_empty:
            return None
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            return None

        window = self._window_from_bounds(polygon.bounds)
        if window is None:
            return None

        data = self.src.read(1, window=window, masked=True)
        if data.size == 0:
            return None

        inside_polygon = self.geometry_mask(
            [self.mapping(polygon)],
            transform=self.src.window_transform(window),
            invert=True,
            out_shape=data.shape,
            all_touched=self.all_touched,
        )
        valid_pixels = inside_polygon & ~np.ma.getmaskarray(data)
        if not np.any(valid_pixels):
            return None

        z_values = data.data[valid_pixels].astype(np.float32, copy=False)
        z_values = z_values * np.float32(self.z_scale) + np.float32(self.z_offset)
        return z_values

    def get_min_max(self, poly, x_offset, y_offset, idx=None):
        stats = self.get_stats(poly, x_offset, y_offset, idx=idx)
        if stats["z_min"] is None or stats["z_max"] is None:
            return DEFAULT_HEIGHT_RANGE
        return _normalize_min_max(stats["z_min"], stats["z_max"])

    def get_stats(self, poly, x_offset, y_offset, idx=None):
        return _summarize_z_values(self._sample_polygon_heights(poly, x_offset, y_offset, idx=idx))


def load_grid_height_sampler(grid_path):
    if not os.path.exists(grid_path):
        logger.error("Height grid file not found: %s", grid_path)
        return None
    return GridHeightSampler(grid_path)


def get_min_max_height(height_sampler, poly, x_offset, y_offset, idx=None):
    if height_sampler is None:
        return DEFAULT_HEIGHT_RANGE
    return height_sampler.get_min_max(poly, x_offset, y_offset, idx=idx)


def get_height_stats(height_sampler, poly, x_offset, y_offset, idx=None):
    if height_sampler is None:
        return _summarize_z_values(None)
    if hasattr(height_sampler, "get_stats"):
        return height_sampler.get_stats(poly, x_offset, y_offset, idx=idx)
    z_min, z_max = get_min_max_height(height_sampler, poly, x_offset, y_offset, idx=idx)
    return _summarize_z_values(np.asarray([z_min, z_max], dtype=np.float32))
