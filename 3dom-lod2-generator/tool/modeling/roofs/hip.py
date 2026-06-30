import os
import subprocess
import sys


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(parent_dir)

from io_utils.exporter import export_polygon_to_txt
from io_utils.importer import import_ply
import modeling.blender_ops as blender_ops
from shapefile.converter import create_mesh_from_polygon


CPP_PATH = "/workspace/3dom-lod2-generator/tool/cpp/build/extrude_skeleton"


def run_executable(exe_path, args=None):
    cmd = [exe_path]
    if args:
        cmd.extend(args)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout, proc.stderr, proc.returncode


def create_hip_roof(base_obj, height, idx, exterior_coords, round_edges=False):
    blender_ops.merge_close_vertices(base_obj)

    txt_path = f"/tmp/input_hip_{idx}.txt"
    out_mesh_path = f"/tmp/hip_{idx}.ply"
    export_polygon_to_txt(base_obj, txt_path)

    base_extrude_height = height
    stdout, stderr, code = run_executable(CPP_PATH, [txt_path, out_mesh_path, "20000.0"])

    if code != 0:
        print("External C++ process failed. Skipping hip roof generation.")
        blender_ops.extrude_faces_z(base_obj, height)
    else:
        try:
            hip_obj = import_ply(out_mesh_path)
            blender_ops.delete_downward_faces(hip_obj)

            hip_height = blender_ops.get_mesh_height(hip_obj)
            base_extrude_height = height - hip_height
            if base_extrude_height < 0:
                base_extrude_height = 1.0

            blender_ops.extrude_faces_z(base_obj, base_extrude_height)
            blender_ops.align_bottom_to_top(hip_obj, base_obj)
            blender_ops.delete_facing_up_faces(base_obj)
            blender_ops.join_meshes(base_obj, hip_obj)
            blender_ops.merge_close_vertices(base_obj)
        except Exception:
            print("Failed importing hip geometry.")
    try:
        if os.path.exists(txt_path):
            os.remove(txt_path)
        if os.path.exists(out_mesh_path):
            os.remove(out_mesh_path)
    except OSError:
        pass

    if round_edges:
        round_obj = create_mesh_from_polygon("round_edge", exterior_coords, [])
        blender_ops.merge_close_vertices(round_obj)
        blender_ops.limited_dissolve_all_faces(round_obj)
        blender_ops.compute_custom_vertex_attribute(round_obj, target_coords=exterior_coords)
        blender_ops.apply_bevel_modifier(round_obj, width=2)
        blender_ops.extrude_faces_z(round_obj, base_extrude_height + 100)
        blender_ops.apply_boolean_intersect(base_obj, round_obj, apply=True)
        blender_ops.triangulate_mesh(base_obj)
        blender_ops.merge_close_vertices(base_obj)
        blender_ops.limited_dissolve_all_faces(base_obj)
        blender_ops.triangulate_mesh(base_obj)

    blender_ops.triangulate_mesh(base_obj)
