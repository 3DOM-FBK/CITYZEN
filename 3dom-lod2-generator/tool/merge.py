#!/usr/bin/env python3
"""
Merge multiple 3D building models into OBJ, PLY, and georeferenced CityJSON.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

import merge_base as base_merge
from shapefile.reader import read_shapefile_polygons


log_level = os.environ.get("LOGLEVEL", "INFO").upper()
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=getattr(logging, log_level),
    format=log_format,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info(f"Merge: Initialisation avec niveau de log: {log_level}")


_PLY_SCALAR_DTYPES = {
    "char": "b",
    "uchar": "B",
    "int8": "b",
    "uint8": "B",
    "short": "h",
    "ushort": "H",
    "int16": "h",
    "uint16": "H",
    "int": "i",
    "uint": "I",
    "int32": "i",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}

CITYJSON_VERTEX_SCALE = float(os.environ.get("CITYZEN_CITYJSON_VERTEX_SCALE", "0.00001"))
CITYJSON_MIN_VERTEX_SCALE = float(os.environ.get("CITYZEN_CITYJSON_MIN_VERTEX_SCALE", "0.000001"))
CITYJSON_EDGE_SCALE_RATIO = float(os.environ.get("CITYZEN_CITYJSON_EDGE_SCALE_RATIO", "0.1"))
CITYJSON_WALL_NORMAL_Z_THRESHOLD = float(os.environ.get("CITYZEN_CITYJSON_WALL_NORMAL_Z_THRESHOLD", "0.25"))
CITYJSON_GROUND_HEIGHT_RATIO = float(os.environ.get("CITYZEN_CITYJSON_GROUND_HEIGHT_RATIO", "0.05"))
CITYJSON_GROUND_HEIGHT_MIN = float(os.environ.get("CITYZEN_CITYJSON_GROUND_HEIGHT_MIN", "0.2"))


def _parse_ply_header(ply_path):
    header_lines = []
    with open(ply_path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            decoded = line.decode("utf-8").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                header_size = f.tell()
                break
        else:
            header_size = 0
    if not header_lines or header_lines[-1] != "end_header":
        raise ValueError(f"PLY header not terminated in {ply_path}")
    return header_lines, header_size


def _read_ascii_ply(ply_path, header_lines):
    vertices = []
    faces = []
    vertex_count = 0
    face_count = 0
    vertex_start = 0
    face_start = 0

    with open(ply_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face"):
            face_count = int(line.split()[-1])
        elif line == "end_header":
            vertex_start = i + 1
            face_start = vertex_start + vertex_count
            break

    for i in range(vertex_start, vertex_start + vertex_count):
        parts = lines[i].strip().split()
        if len(parts) >= 3:
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])

    for i in range(face_start, face_start + face_count):
        parts = lines[i].strip().split()
        if len(parts) < 4:
            continue
        vertex_total = int(parts[0])
        indices = [int(parts[j]) for j in range(1, 1 + vertex_total)]
        if vertex_total == 3:
            faces.append(indices)
        elif vertex_total == 4:
            faces.append([indices[0], indices[1], indices[2]])
            faces.append([indices[0], indices[2], indices[3]])
    return np.asarray(vertices, dtype=np.float64), faces


def _read_binary_ply(ply_path, header_lines, header_size):
    endianness = "<"
    if any("format binary_big_endian" in line for line in header_lines):
        endianness = ">"

    vertex_count = 0
    face_count = 0
    vertex_properties = []
    face_list_types = None
    current_element = None

    for line in header_lines:
        if line.startswith("element vertex"):
            current_element = "vertex"
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face"):
            current_element = "face"
            face_count = int(line.split()[-1])
        elif line.startswith("element "):
            current_element = None
        elif line.startswith("property") and current_element == "vertex":
            parts = line.split()
            if len(parts) == 3:
                _, dtype_name, prop_name = parts
                vertex_properties.append((prop_name, dtype_name))
        elif line.startswith("property list") and current_element == "face":
            parts = line.split()
            if len(parts) == 5 and parts[-1] == "vertex_indices":
                face_list_types = (parts[2], parts[3])

    if face_list_types is None:
        raise ValueError(f"Unsupported face layout in {ply_path}")

    vertex_struct = endianness + "".join(_PLY_SCALAR_DTYPES[dtype_name] for _, dtype_name in vertex_properties)
    vertex_size = struct.calcsize(vertex_struct)
    count_type, item_type = face_list_types
    count_struct = endianness + _PLY_SCALAR_DTYPES[count_type]
    item_struct = endianness + _PLY_SCALAR_DTYPES[item_type]
    count_size = struct.calcsize(count_struct)
    item_size = struct.calcsize(item_struct)

    vertices = np.zeros((vertex_count, 3), dtype=np.float64)
    faces = []

    with open(ply_path, "rb") as f:
        f.seek(header_size)

        for i in range(vertex_count):
            data = f.read(vertex_size)
            if len(data) != vertex_size:
                raise ValueError(f"Unexpected EOF while reading vertices from {ply_path}")
            values = struct.unpack(vertex_struct, data)
            value_map = {name: value for (name, _), value in zip(vertex_properties, values)}
            vertices[i, 0] = float(value_map.get("x", 0.0))
            vertices[i, 1] = float(value_map.get("y", 0.0))
            vertices[i, 2] = float(value_map.get("z", 0.0))

        for _ in range(face_count):
            raw_count = f.read(count_size)
            if len(raw_count) != count_size:
                raise ValueError(f"Unexpected EOF while reading faces from {ply_path}")
            index_count = struct.unpack(count_struct, raw_count)[0]
            raw_indices = f.read(item_size * index_count)
            if len(raw_indices) != item_size * index_count:
                raise ValueError(f"Unexpected EOF while reading face indices from {ply_path}")
            indices = list(struct.unpack(endianness + (_PLY_SCALAR_DTYPES[item_type] * index_count), raw_indices))
            if index_count == 3:
                faces.append(indices)
            elif index_count == 4:
                faces.append([indices[0], indices[1], indices[2]])
                faces.append([indices[0], indices[2], indices[3]])

    return vertices, faces


def read_ply_file(ply_path):
    header_lines, header_size = _parse_ply_header(ply_path)
    if any("format ascii" in line for line in header_lines):
        return _read_ascii_ply(ply_path, header_lines)
    if any("format binary_" in line for line in header_lines):
        return _read_binary_ply(ply_path, header_lines, header_size)
    raise ValueError(f"Unsupported PLY format in {ply_path}")


def _load_crs_metadata(source_vector=None, source_raster=None):
    crs = None
    if source_vector and Path(source_vector).exists():
        try:
            crs = gpd.read_file(source_vector, rows=1).crs
        except Exception as exc:
            logger.warning(f"Could not read CRS from vector {source_vector}: {exc}")
    if crs is None and source_raster and Path(source_raster).exists():
        try:
            with rasterio.open(source_raster) as src:
                crs = src.crs
        except Exception as exc:
            logger.warning(f"Could not read CRS from raster {source_raster}: {exc}")
    if crs is None:
        return None

    epsg = None
    try:
        epsg = crs.to_epsg()
    except Exception:
        epsg = None
    if epsg is None:
        return {}
    return {
        "epsg": epsg,
        "reference_system": f"EPSG:{epsg}",
        "reference_system_uri": f"https://www.opengis.net/def/crs/EPSG/0/{epsg}",
    }


def _load_building_attributes(source_vector):
    if not source_vector or not Path(source_vector).exists():
        return {}

    polygons, _ = read_shapefile_polygons(source_vector)
    attributes = {}
    for index, polygon in enumerate(polygons):
        attributes[f"out_{index}"] = {
            "roof_type": polygon.get("roof"),
            "height": polygon.get("height"),
        }
    return attributes


def _face_normal(vertices, face):
    if len(face) < 3:
        return None
    v0 = vertices[int(face[0])]
    v1 = vertices[int(face[1])]
    vertex_2 = vertices[int(face[2])]
    normal = np.cross(v1 - v0, vertex_2 - v0)
    norm = np.linalg.norm(normal)
    if norm <= 1e-12:
        return None
    return normal / norm


def _semantic_surface_templates():
    return [
        {"type": "GroundSurface"},
        {"type": "WallSurface", "on_footprint_edge": True},
        {"type": "RoofSurface"},
    ]


def _classify_face_semantic(mesh_vertices, local_face, min_z, max_z):
    face_vertices = mesh_vertices[np.asarray(local_face, dtype=np.int64)]
    centroid_z = float(face_vertices[:, 2].mean())
    normal = _face_normal(mesh_vertices, local_face)
    mesh_height = max(float(max_z - min_z), 0.0)
    ground_tol = max(CITYJSON_GROUND_HEIGHT_MIN, mesh_height * CITYJSON_GROUND_HEIGHT_RATIO)

    if normal is not None and abs(float(normal[2])) >= 0.5 and centroid_z <= (min_z + ground_tol):
        return 0
    if normal is None:
        return 1 if centroid_z > (min_z + ground_tol) else 0
    if abs(float(normal[2])) <= CITYJSON_WALL_NORMAL_Z_THRESHOLD:
        return 1
    return 2


def _compute_cityjson_transform(all_vertices):
    combined_vertices = np.vstack([np.asarray(vertices, dtype=np.float64) for vertices in all_vertices if len(vertices)])
    min_xyz = combined_vertices.min(axis=0)
    max_xyz = combined_vertices.max(axis=0)

    min_nonzero_edge = None
    for vertices in all_vertices:
        vertices = np.asarray(vertices, dtype=np.float64)
        if len(vertices) < 2:
            continue
        edge_lengths = np.linalg.norm(vertices[1:] - vertices[:-1], axis=1)
        edge_lengths = edge_lengths[edge_lengths > 1e-9]
        if edge_lengths.size == 0:
            continue
        current_min = float(edge_lengths.min())
        min_nonzero_edge = current_min if min_nonzero_edge is None else min(min_nonzero_edge, current_min)

    scale_value = CITYJSON_VERTEX_SCALE
    if min_nonzero_edge is not None:
        scale_value = min(scale_value, max(CITYJSON_MIN_VERTEX_SCALE, min_nonzero_edge * CITYJSON_EDGE_SCALE_RATIO))

    transform = {
        "scale": [scale_value, scale_value, scale_value],
        "translate": [float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])],
    }
    return transform, min_xyz, max_xyz


def _quantize_vertices(vertices, transform):
    vertices = np.asarray(vertices, dtype=np.float64)
    scale = np.asarray(transform["scale"], dtype=np.float64)
    translate = np.asarray(transform["translate"], dtype=np.float64)
    quantized = np.rint((vertices - translate) / scale).astype(np.int64)
    return quantized


def _deduplicate_consecutive_indices(face):
    deduplicated = []
    for index in face:
        index = int(index)
        if deduplicated and deduplicated[-1] == index:
            continue
        deduplicated.append(index)
    if len(deduplicated) > 1 and deduplicated[0] == deduplicated[-1]:
        deduplicated.pop()
    return deduplicated


def _polygon_area_3d(vertices, face):
    if len(face) < 3:
        return 0.0
    points = np.asarray([vertices[index] for index in face], dtype=np.float64)
    origin = points[0]
    area_vector = np.zeros(3, dtype=np.float64)
    for i in range(1, len(points) - 1):
        area_vector += np.cross(points[i] - origin, points[i + 1] - origin)
    return float(np.linalg.norm(area_vector) * 0.5)


def _clean_quantized_mesh(mesh_vertices, mesh_faces, transform):
    quantized_vertices = _quantize_vertices(mesh_vertices, transform)

    unique_vertex_map = {}
    unique_vertices = []
    remap = {}
    for vertex_index, quantized_vertex in enumerate(quantized_vertices):
        key = tuple(int(value) for value in quantized_vertex.tolist())
        mapped_index = unique_vertex_map.get(key)
        if mapped_index is None:
            mapped_index = len(unique_vertices)
            unique_vertex_map[key] = mapped_index
            unique_vertices.append(list(key))
        remap[vertex_index] = mapped_index

    cleaned_faces = []
    seen_faces = set()
    cleaned_vertex_array = np.asarray(unique_vertices, dtype=np.float64) if unique_vertices else np.empty((0, 3), dtype=np.float64)
    for face in mesh_faces:
        remapped_face = _deduplicate_consecutive_indices(remap.get(int(index), int(index)) for index in face)
        if len(remapped_face) < 3 or len(set(remapped_face)) < 3:
            continue
        if _polygon_area_3d(cleaned_vertex_array, remapped_face) <= 0.0:
            continue

        canonical_face = tuple(sorted(remapped_face))
        if canonical_face in seen_faces:
            continue
        seen_faces.add(canonical_face)
        cleaned_faces.append(remapped_face)

    return unique_vertices, cleaned_faces


def _cityjson_from_meshes(all_vertices, all_faces, building_names, building_attributes, crs_metadata):
    vertices = []
    cityobjects = {}
    transform, min_xyz, max_xyz = _compute_cityjson_transform(all_vertices)

    for mesh_index, (mesh_vertices, mesh_faces, building_name) in enumerate(zip(all_vertices, all_faces, building_names)):
        mesh_vertices = np.asarray(mesh_vertices, dtype=np.float64)
        if mesh_vertices.size == 0:
            continue

        start_index = len(vertices)
        quantized_vertices, cleaned_faces = _clean_quantized_mesh(mesh_vertices, mesh_faces, transform)
        if not quantized_vertices or not cleaned_faces:
            logger.warning("Skipping CityJSON export for %s because no valid cleaned faces remain", building_name)
            continue
        vertices.extend(quantized_vertices)

        quantized_vertex_array = np.asarray(quantized_vertices, dtype=np.float64)
        surface_boundaries = []
        semantic_values = []
        mesh_min_z = float(quantized_vertex_array[:, 2].min())
        mesh_max_z = float(quantized_vertex_array[:, 2].max())
        for local_face in cleaned_faces:
            global_face = [local_index + start_index for local_index in local_face]
            surface_boundaries.append([global_face])
            semantic_values.append(_classify_face_semantic(quantized_vertex_array, local_face, mesh_min_z, mesh_max_z))

        attributes = {
            "source_mesh": building_name,
        }
        attributes.update(
            {
                key: value
                for key, value in building_attributes.get(building_name, {}).items()
                if value is not None
            }
        )

        cityobjects[building_name] = {
            "type": "Building",
            "attributes": attributes,
            "geometry": [
                {
                    "type": "MultiSurface",
                    "lod": "2.2",
                    "boundaries": surface_boundaries,
                    "semantics": {
                        "surfaces": _semantic_surface_templates(),
                        "values": semantic_values,
                    },
                }
            ],
        }

    metadata = {}
    if np.all(np.isfinite(min_xyz)) and np.all(np.isfinite(max_xyz)):
        metadata["geographicalExtent"] = [
            float(min_xyz[0]),
            float(min_xyz[1]),
            float(min_xyz[2]),
            float(max_xyz[0]),
            float(max_xyz[1]),
            float(max_xyz[2]),
        ]
    if crs_metadata.get("reference_system"):
        metadata["referenceSystem"] = crs_metadata["reference_system"]
    if crs_metadata.get("reference_system_uri"):
        metadata["referenceSystemURI"] = crs_metadata["reference_system_uri"]

    cityjson = {
        "type": "CityJSON",
        "version": "2.0",
        "CityObjects": cityobjects,
        "vertices": vertices,
        "transform": transform,
    }
    if metadata:
        cityjson["metadata"] = metadata
    return cityjson


def write_cityjson_file(output_path, all_vertices, all_faces, building_names, building_attributes=None, crs_metadata=None):
    cityjson = _cityjson_from_meshes(
        all_vertices=all_vertices,
        all_faces=all_faces,
        building_names=building_names,
        building_attributes=building_attributes or {},
        crs_metadata=crs_metadata or {},
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cityjson, f, indent=2)
        f.write("\n")


def merge_3d_models(models_dir, output_path, ply_only=False, source_vector=None, source_raster=None, write_cityjson=True):
    models_dir = Path(models_dir)
    output_path = Path(output_path)

    ply_files = list(models_dir.glob("*.ply"))
    if not ply_files:
        logger.warning(f"No PLY files found in {models_dir}")
        return False

    logger.info(f"Found {len(ply_files)} PLY files to merge")

    all_vertices = []
    all_faces = []
    cityjson_faces = []
    building_names = []
    vertex_offset = 0

    for ply_file in sorted(ply_files):
        logger.debug(f"Processing {ply_file.name}")
        try:
            vertices, faces = read_ply_file(ply_file)
            if len(vertices) == 0:
                logger.warning(f"No vertices found in {ply_file.name}")
                continue

            adjusted_faces = []
            for face in faces:
                adjusted_faces.append([idx + vertex_offset for idx in face])

            all_vertices.append(vertices)
            all_faces.append(adjusted_faces)
            cityjson_faces.append([list(face) for face in faces])
            building_names.append(ply_file.stem)
            vertex_offset += len(vertices)
        except Exception as exc:
            logger.error(f"Error processing {ply_file.name}: {exc}")
            continue

    if not all_vertices:
        logger.error("No valid models found to merge")
        return False

    combined_vertices = np.vstack(all_vertices)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ply_only:
        logger.info(f"Writing merged OBJ model to {output_path}")
        base_merge.write_obj_file(output_path, combined_vertices, all_faces, building_names)

    ply_output = output_path.with_suffix(".ply")
    logger.info(f"Writing merged PLY model to {ply_output}")
    base_merge.write_ply_file(ply_output, combined_vertices, all_faces, building_names)

    if write_cityjson:
        cityjson_output = output_path.with_suffix(".city.json")
        crs_metadata = _load_crs_metadata(source_vector=source_vector, source_raster=source_raster)
        building_attributes = _load_building_attributes(source_vector)
        logger.info(f"Writing merged CityJSON model to {cityjson_output}")
        write_cityjson_file(
            cityjson_output,
            all_vertices=all_vertices,
            all_faces=cityjson_faces,
            building_names=building_names,
            building_attributes=building_attributes,
            crs_metadata=crs_metadata,
        )

    total_vertices = len(combined_vertices)
    total_faces = sum(len(faces) for faces in all_faces)
    logger.info("Merged model created successfully:")
    logger.info(f"  - {len(building_names)} buildings")
    logger.info(f"  - {total_vertices} total vertices")
    logger.info(f"  - {total_faces} total faces")
    if not ply_only:
        logger.info(f"  - OBJ Output: {output_path}")
    logger.info(f"  - PLY Output: {ply_output}")
    if write_cityjson:
        logger.info(f"  - CityJSON Output: {output_path.with_suffix('.city.json')}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge 3D building models into OBJ, PLY, and CityJSON")
    parser.add_argument("--models_dir", required=True, help="Directory containing PLY model files")
    parser.add_argument("--output", required=True, help="Output OBJ file path")
    parser.add_argument("--ply-only", action="store_true", help="Only generate PLY output (not OBJ)")
    parser.add_argument("--source_vector", help="Optional classified vector file for CRS and building attributes")
    parser.add_argument("--source_raster", help="Optional DSM raster for CRS fallback")
    parser.add_argument("--no-cityjson", action="store_true", help="Disable CityJSON export")
    args = parser.parse_args()

    if not os.path.exists(args.models_dir):
        logger.error(f"Models directory not found: {args.models_dir}")
        return 1

    success = merge_3d_models(
        models_dir=args.models_dir,
        output_path=args.output,
        ply_only=args.ply_only,
        source_vector=args.source_vector,
        source_raster=args.source_raster,
        write_cityjson=not args.no_cityjson,
    )
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
