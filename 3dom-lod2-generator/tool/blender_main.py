import argparse
import os
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import blender_common as base_blender_main
import modeling.pointcloud_ops as pointcloud_ops
from modeling.roofs.gabled_L import create_gabled_L_roof
from modeling.roofs.hip import create_hip_roof
from modeling.roofs.pyramid import create_pyramid_roof
from shapefile.reader import read_shapefile_polygons


SUPPORTED_ROOF_TYPES = {"flat", "gabled", "gabled-L", "hip", "pyramid"}
CLASSIFIER_TO_3DOM_ROOF = {
    "flat": "flat",
    "gable": "gabled",
    "hip": "hip",
    "l-shaped": "gabled-L",
    "pyramid": "pyramid",
}

COMPLEX_CONCAVITY_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_CONCAVITY_THRESHOLD", "0.90"))
COMPLEX_STRONG_CONCAVITY_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_STRONG_CONCAVITY_THRESHOLD", "0.82"))
COMPLEX_SUPPORTED_PROB_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_SUPPORTED_PROB_THRESHOLD", "0.35"))
COMPLEX_SUPPORTED_MARGIN_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_SUPPORTED_MARGIN_THRESHOLD", "0.08"))
COMPLEX_FLAT_RELIEF_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_FLAT_RELIEF_THRESHOLD", "1.25"))
COMPLEX_ELONGATED_ASPECT_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_ELONGATED_ASPECT_THRESHOLD", "1.6"))
COMPLEX_COMPACT_ASPECT_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_COMPACT_ASPECT_THRESHOLD", "1.25"))
COMPLEX_FLAT_PROB_THRESHOLD = float(os.environ.get("CITYZEN_COMPLEX_FLAT_PROB_THRESHOLD", "0.45"))
MESH_SANITY_BBOX_MARGIN = float(os.environ.get("CITYZEN_MESH_SANITY_BBOX_MARGIN", "1.0"))
MESH_SANITY_MAX_EXTENT_RATIO = float(os.environ.get("CITYZEN_MESH_SANITY_MAX_EXTENT_RATIO", "1.45"))
MESH_SANITY_MAX_HEIGHT_RATIO = float(os.environ.get("CITYZEN_MESH_SANITY_MAX_HEIGHT_RATIO", "1.75"))
MESH_SANITY_MAX_HEIGHT_BUFFER = float(os.environ.get("CITYZEN_MESH_SANITY_MAX_HEIGHT_BUFFER", "4.0"))
SAFE_RETRY_GABLED_ASPECT_THRESHOLD = float(os.environ.get("CITYZEN_SAFE_RETRY_GABLED_ASPECT_THRESHOLD", "1.35"))


def parse_args():
    argv = sys.argv
    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Process 3D buildings from vector footprints in Blender.")
    parser.add_argument("-i", "--input_shapefile", type=str, required=True, help="Path to the input vector file.")
    parser.add_argument("-o", "--output_folder", type=str, required=True, help="Folder where the generated meshes will be saved.")
    parser.add_argument("--ortho_name", type=str, default=None, help="Name of the orthophoto (used for creating subdirectory)")
    parser.add_argument("-r", "--round_edges", action="store_true", help="Apply rounding (bevel) to roof edges.")
    parser.add_argument(
        "--export_format",
        type=str,
        default="ply",
        choices=["ply", "obj"],
        help="File format to export the resulting mesh (default: ply).",
    )
    parser.add_argument("--dsm_grid", type=str, help="Precomputed calibrated DSM height grid NPZ file")
    parser.add_argument("--dsm_raster", type=str, help="DSM raster TIF file used directly for per-building height queries")
    parser.add_argument(
        "--ndsm_height_scale",
        type=float,
        default=float(os.environ.get("CITYZEN_NDSM_HEIGHT_SCALE", "1.0")),
        help="Multiply normalized nDSM values by this factor before roof extrusion",
    )
    parser.add_argument(
        "--ndsm_height_offset",
        type=float,
        default=float(os.environ.get("CITYZEN_NDSM_HEIGHT_OFFSET", "0.0")),
        help="Add this offset to nDSM-derived heights before roof extrusion",
    )
    return parser.parse_args(argv)


def _footprint_xy_ring(poly):
    exterior = poly.get("exterior", [])
    coords = [(float(coord[0]), float(coord[1])) for coord in exterior if len(coord) >= 2]
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords


def _polygon_area(coords):
    if len(coords) < 3:
        return 0.0
    area = 0.0
    for i, (x0, y0) in enumerate(coords):
        x1, y1 = coords[(i + 1) % len(coords)]
        area += (x0 * y1) - (x1 * y0)
    return abs(area) * 0.5


def _convex_hull(points):
    unique_points = sorted(set(points))
    if len(unique_points) <= 1:
        return unique_points

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])

    lower = []
    for point in unique_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _footprint_shape_metrics(poly):
    coords = _footprint_xy_ring(poly)
    if len(coords) < 3:
        return {
            "vertex_count": len(coords),
            "area": 0.0,
            "convexity_ratio": 1.0,
            "aspect_ratio": 1.0,
        }

    area = _polygon_area(coords)
    hull = _convex_hull(coords)
    hull_area = _polygon_area(hull) if len(hull) >= 3 else area
    convexity_ratio = (area / hull_area) if hull_area > 1e-12 else 1.0

    points = np.asarray(coords, dtype=np.float64)
    centered = points - np.mean(points, axis=0, keepdims=True)
    if points.shape[0] >= 3 and np.any(centered):
        covariance = np.cov(centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))
        if eigenvalues[-1] <= 1e-12:
            aspect_ratio = 1.0
        elif eigenvalues[0] <= 1e-12:
            aspect_ratio = float("inf")
        else:
            aspect_ratio = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    else:
        aspect_ratio = 1.0

    return {
        "vertex_count": len(coords),
        "area": float(area),
        "convexity_ratio": float(convexity_ratio),
        "aspect_ratio": float(aspect_ratio),
    }


def _footprint_bbox(poly):
    coords = _footprint_xy_ring(poly)
    if not coords:
        return None
    points = np.asarray(coords, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return mins, maxs


def _mesh_vertex_array(obj):
    vertices = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    if not vertices:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray([[vertex.x, vertex.y, vertex.z] for vertex in vertices], dtype=np.float64)


def _mesh_is_plausible(obj, poly, expected_height, idx=None):
    vertices = _mesh_vertex_array(obj)
    if vertices.shape[0] == 0:
        return False, "empty_mesh"

    bbox = _footprint_bbox(poly)
    if bbox is None:
        return True, "no_bbox"

    footprint_min, footprint_max = bbox
    mesh_min = vertices[:, :2].min(axis=0)
    mesh_max = vertices[:, :2].max(axis=0)
    footprint_extent = np.maximum(footprint_max - footprint_min, 1e-6)
    mesh_extent = mesh_max - mesh_min
    allowed_extent = np.maximum(
        footprint_extent * MESH_SANITY_MAX_EXTENT_RATIO,
        footprint_extent + (2.0 * MESH_SANITY_BBOX_MARGIN),
    )

    min_allowed = footprint_min - MESH_SANITY_BBOX_MARGIN
    max_allowed = footprint_max + MESH_SANITY_BBOX_MARGIN
    if np.any(mesh_min < min_allowed) or np.any(mesh_max > max_allowed):
        if np.any(mesh_extent > allowed_extent):
            return False, "xy_extent"

    mesh_height = float(vertices[:, 2].max() - vertices[:, 2].min())
    max_reasonable_height = max(
        float(expected_height) * MESH_SANITY_MAX_HEIGHT_RATIO,
        float(expected_height) + MESH_SANITY_MAX_HEIGHT_BUFFER,
    )
    if mesh_height > max_reasonable_height:
        return False, "height_spike"

    return True, "ok"


def _select_safe_retry_roof(current_roof_type, poly, raw_relief):
    shape_metrics = _footprint_shape_metrics(poly)
    aspect_ratio = shape_metrics["aspect_ratio"]

    if raw_relief is not None and raw_relief <= COMPLEX_FLAT_RELIEF_THRESHOLD:
        return "flat"
    if current_roof_type == "gabled-L":
        return "gabled" if aspect_ratio >= SAFE_RETRY_GABLED_ASPECT_THRESHOLD else "hip"
    if current_roof_type == "hip":
        return "gabled" if aspect_ratio >= SAFE_RETRY_GABLED_ASPECT_THRESHOLD else "flat"
    if current_roof_type == "pyramid":
        return "hip" if aspect_ratio <= COMPLEX_COMPACT_ASPECT_THRESHOLD else "gabled"
    if current_roof_type == "gabled":
        return "hip" if aspect_ratio <= COMPLEX_COMPACT_ASPECT_THRESHOLD else "flat"
    return "flat"


def _create_base_object(obj_name, poly, absolute_z_min):
    obj = base_blender_main.create_mesh_from_polygon(obj_name, poly["exterior"], poly["holes"])
    base_blender_main.blender_ops.flatten_mesh_to_z(obj, absolute_z_min)
    return obj


def _build_roof_dispatch(obj, idx, poly, args):
    return {
        "flat": lambda: base_blender_main.create_flat_roof(
            obj, poly["height"], poly["exterior"], round_edges=args.round_edges
        ),
        "gabled": lambda: base_blender_main.create_gabled_roof(
            obj, poly["height"], poly["exterior"], round_edges=args.round_edges
        ),
        "gabled-L": lambda: create_gabled_L_roof(
            obj, poly["height"], idx, poly["exterior"], round_edges=args.round_edges
        ),
        "hip": lambda: create_hip_roof(
            obj, poly["height"], idx, poly["exterior"], round_edges=args.round_edges
        ),
        "pyramid": lambda: create_pyramid_roof(
            obj, poly["height"], idx, poly["exterior"], round_edges=args.round_edges
        ),
    }


def _best_supported_roof_from_probabilities(poly):
    probabilities = poly.get("roof_probabilities") or {}
    ranked = []
    for classifier_label, mapped_label in CLASSIFIER_TO_3DOM_ROOF.items():
        probability = float(probabilities.get(classifier_label, 0.0) or 0.0)
        ranked.append((probability, classifier_label, mapped_label))
    ranked.sort(reverse=True)
    if not ranked:
        return None, 0.0, 0.0
    best_probability, _, best_label = ranked[0]
    second_probability = ranked[1][0] if len(ranked) > 1 else 0.0
    return best_label, float(best_probability), float(second_probability)


def _resolve_fallback_roof_type(poly, raw_relief, idx=None):
    shape_metrics = _footprint_shape_metrics(poly)
    convexity_ratio = shape_metrics["convexity_ratio"]
    aspect_ratio = shape_metrics["aspect_ratio"]
    best_supported_roof, best_probability, second_probability = _best_supported_roof_from_probabilities(poly)

    if convexity_ratio < COMPLEX_STRONG_CONCAVITY_THRESHOLD:
        return "gabled-L", "strong_concavity"

    if raw_relief is not None and raw_relief <= COMPLEX_FLAT_RELIEF_THRESHOLD:
        if best_supported_roof == "flat" or best_probability >= COMPLEX_FLAT_PROB_THRESHOLD:
            return "flat", "low_relief"

    if best_supported_roof == "gabled-L" and convexity_ratio < COMPLEX_CONCAVITY_THRESHOLD:
        return "gabled-L", "classifier_l_shape"

    if best_supported_roof is not None:
        probability_margin = best_probability - second_probability
        if (
            best_probability >= COMPLEX_SUPPORTED_PROB_THRESHOLD
            and probability_margin >= COMPLEX_SUPPORTED_MARGIN_THRESHOLD
        ):
            if best_supported_roof == "flat" and raw_relief is not None and raw_relief > COMPLEX_FLAT_RELIEF_THRESHOLD:
                pass
            else:
                return best_supported_roof, "supported_probability"

    if convexity_ratio < COMPLEX_CONCAVITY_THRESHOLD:
        return "gabled-L", "concavity"

    if raw_relief is not None and raw_relief <= COMPLEX_FLAT_RELIEF_THRESHOLD:
        return "flat", "fallback_low_relief"

    if aspect_ratio >= COMPLEX_ELONGATED_ASPECT_THRESHOLD:
        return "gabled", "elongated_footprint"

    if best_supported_roof == "pyramid" and aspect_ratio <= COMPLEX_COMPACT_ASPECT_THRESHOLD:
        return "pyramid", "classifier_compact"

    if best_supported_roof == "hip":
        return "hip", "classifier_compact"

    if aspect_ratio <= COMPLEX_COMPACT_ASPECT_THRESHOLD:
        return "hip", "compact_footprint"

    return "gabled", "default_fallback"


def _footprint_ground_z(poly):
    exterior = poly.get("exterior", [])
    if not exterior:
        return None

    z_values = [coord[2] for coord in exterior if len(coord) >= 3]
    if not z_values:
        return None

    z_min = min(z_values)
    z_max = max(z_values)
    if abs(z_max - z_min) > 1e-6:
        return None
    return z_min


def export_and_shift_mesh(obj, i, x_offset, y_offset, output_folder, export_format="ply"):
    assert export_format in ["ply", "obj"], "Unsupported export format"

    out_path = os.path.join(output_folder, f"out_{i}.{export_format}")

    if export_format == "ply":
        obj.location.x += x_offset
        obj.location.y += y_offset
        try:
            base_blender_main.export_mesh_ply(out_path, obj, False)
        finally:
            obj.location.x -= x_offset
            obj.location.y -= y_offset
    else:
        tmp_path = os.path.join(output_folder, f".tmp_out_{i}.ply")
        base_blender_main.export_mesh_ply(tmp_path, obj, True)
        base_blender_main.apply_global_shift(tmp_path, out_path, x_offset, y_offset)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    base_blender_main.blender_ops.clear_blender_scene()
    base_blender_main.logger.debug(f"----> Saved mesh to: {out_path}")


def process_roofs(polygons_to_process, x_offset, y_offset, height_sampler, args, force_roof_type=None):
    failed_indices = []
    step_start = time.perf_counter()

    for i, poly in enumerate(polygons_to_process):
        idx = poly["index"] if "index" in poly else i
        obj_name = f"Building_{idx}"
        base_blender_main.logger.debug(f"--> Processing {obj_name}...")

        z_min, z_max = pointcloud_ops.get_min_max_height(height_sampler, poly, x_offset, y_offset, idx)
        height_stats = pointcloud_ops.get_height_stats(height_sampler, poly, x_offset, y_offset, idx)
        raw_relief = height_stats.get("relief")
        footprint_ground_z = _footprint_ground_z(poly)

        if z_max is not None:
            base_blender_main.logger.debug(f"Highest point: {z_max}")
            base_blender_main.logger.debug(f"Lowest point: {z_min}")
        else:
            base_blender_main.logger.debug("No points found in the bounding box.")

        if (
            footprint_ground_z is not None
            and z_max is not None
            and abs(footprint_ground_z) > 100
            and abs(z_max) < 100
        ):
            absolute_z_min = footprint_ground_z + z_min
            absolute_z_max = footprint_ground_z + z_max
            base_blender_main.logger.debug(
                f"Building {idx}: using footprint ground Z {footprint_ground_z:.3f} with "
                f"relative nDSM points [{z_min:.3f}, {z_max:.3f}]"
            )
        else:
            absolute_z_min = z_min
            absolute_z_max = z_max

        shapefile_height = poly.get("height")
        dsm_computed_height = absolute_z_max - absolute_z_min

        base_blender_main.logger.debug(
            f"Building {idx}: DSM height calculation: z_min={absolute_z_min}, "
            f"z_max={absolute_z_max}, computed_height={dsm_computed_height}"
        )
        base_blender_main.logger.debug(f"Building {idx}: Shapefile height: {shapefile_height}")

        if shapefile_height is not None and shapefile_height > 0 and shapefile_height < 200:
            final_height = shapefile_height
            base_blender_main.logger.debug(f"Building {idx}: Using shapefile height: {final_height}")
        else:
            final_height = dsm_computed_height
            base_blender_main.logger.debug(f"Building {idx}: Using computed DSM height: {final_height}")

        poly["height"] = final_height

        roof_type = force_roof_type if force_roof_type else poly.get("roof")
        if roof_type not in SUPPORTED_ROOF_TYPES:
            roof_type, resolution_reason = _resolve_fallback_roof_type(poly, raw_relief, idx=idx)
            poly["resolved_roof"] = roof_type
            base_blender_main.logger.debug(
                "Building %s: resolved roof '%s' from source '%s' using %s "
                "(raw_relief=%s, probabilities=%s)",
                idx,
                roof_type,
                poly.get("roof_source", poly.get("roof")),
                resolution_reason,
                f"{raw_relief:.3f}" if raw_relief is not None else "n/a",
                poly.get("roof_probabilities", {}),
            )
        base_blender_main.logger.debug(
            f"Building {idx}: roof_type='{roof_type}', height={poly.get('height', 'N/A')}"
        )

        candidate_roofs = [roof_type]
        safe_retry_roof = _select_safe_retry_roof(roof_type, poly, raw_relief)
        if safe_retry_roof not in candidate_roofs:
            candidate_roofs.append(safe_retry_roof)

        successful_obj = None
        for candidate_roof in candidate_roofs:
            # Roof generators can leave helper meshes in the scene; start each
            # attempt from a clean Blender state so retries cannot inherit them.
            base_blender_main.blender_ops.clear_blender_scene()
            obj = _create_base_object(obj_name, poly, absolute_z_min)
            roof_dispatch = _build_roof_dispatch(obj, idx, poly, args)
            if candidate_roof not in roof_dispatch:
                base_blender_main.blender_ops.clear_blender_scene()
                continue

            roof_dispatch[candidate_roof]()

            if len(obj.data.vertices) == 0:
                base_blender_main.blender_ops.clear_blender_scene()
                continue

            mesh_ok, mesh_reason = _mesh_is_plausible(obj, poly, poly["height"], idx=idx)
            if mesh_ok:
                if candidate_roof != roof_type:
                    base_blender_main.logger.warning(
                        "Building %s: roof '%s' produced an implausible mesh earlier, using safer fallback '%s'",
                        idx,
                        roof_type,
                        candidate_roof,
                    )
                poly["generated_roof"] = candidate_roof
                successful_obj = obj
                break

            base_blender_main.logger.warning(
                "Building %s: rejecting roof '%s' because generated mesh failed sanity check (%s)",
                idx,
                candidate_roof,
                mesh_reason,
            )
            base_blender_main.blender_ops.clear_blender_scene()

        if successful_obj is None:
            failed_indices.append(idx)
            continue

        export_and_shift_mesh(successful_obj, idx, x_offset, y_offset, args.output_folder, args.export_format)

        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - step_start
            base_blender_main.logger.info(
                "Processed %d/%d buildings in %.2f seconds",
                i + 1,
                len(polygons_to_process),
                elapsed,
            )

    return failed_indices


if __name__ == "__main__":
    args = parse_args()
    start = time.perf_counter()

    base_blender_main.logger.info("Read vector footprints...")
    polygons, (x_offset, y_offset) = read_shapefile_polygons(args.input_shapefile)

    if args.ortho_name:
        ortho_output_dir = os.path.join(args.output_folder, args.ortho_name)
        os.makedirs(ortho_output_dir, exist_ok=True)
        base_blender_main.logger.info(f"Created output directory: {ortho_output_dir}")
        args.output_folder = ortho_output_dir

    base_blender_main.logger.info("--> Read point cloud data...")
    if args.dsm_grid:
        base_blender_main.logger.info("Using precomputed DSM height grid...")
        height_sampler = pointcloud_ops.load_grid_height_sampler(args.dsm_grid)
    elif args.dsm_raster:
        base_blender_main.logger.info("Using DSM raster height sampler...")
        height_sampler = pointcloud_ops.RasterHeightSampler(
            args.dsm_raster,
            z_scale=args.ndsm_height_scale,
            z_offset=args.ndsm_height_offset,
        )
    else:
        base_blender_main.logger.info("No height source provided (neither DSM grid nor DSM raster)")
        height_sampler = None

    for i, poly in enumerate(polygons):
        poly["index"] = i

    try:
        failed_idxs = process_roofs(polygons, x_offset, y_offset, height_sampler, args)

        if failed_idxs:
            base_blender_main.logger.info(f"\n---> Retry su {len(failed_idxs)} edifici con tetto flat")
            retry_polygons = [polygons[i] for i in failed_idxs]
            process_roofs(
                retry_polygons,
                x_offset,
                y_offset,
                height_sampler,
                args,
                force_roof_type="flat",
            )
    finally:
        if height_sampler is not None:
            height_sampler.close()

    end = time.perf_counter()
    base_blender_main.logger.info(f"Execution time: {end - start:.4f} seconds")
