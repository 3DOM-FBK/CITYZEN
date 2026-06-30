import bpy
import bmesh
import os
import sys
import trimesh
import numpy as np


#######################################################
# Adds the root project in the Python path
#######################################################
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#######################################################

import modeling.blender_ops as blender_ops

### function: export_mesh_ply ###
def export_mesh_ply(filepath, obj=None, use_ascii=False):
    """
    Exports a mesh to PLY format (binary or ASCII) using Blender 4.4's wm.ply_export operator.

    Args:
        filepath (str): Full path to the .ply file to be created.
        obj (bpy.types.Object, optional): If provided, only this object will be exported;
                                          otherwise, the entire scene is exported.
        use_ascii (bool): True for ASCII format, False for binary format.
    """
    if obj:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        use_selection = True
    else:
        use_selection = False

    bpy.ops.wm.ply_export(
        filepath=filepath,
        export_selected_objects=use_selection,
        ascii_format=use_ascii
    )


### function: export_polygon_to_txt ###
def export_polygon_to_txt(obj, filepath):
    exterior_indices, holes_indices = blender_ops.get_exterior_and_hole_loops(obj)

    mesh = obj.data
    verts = mesh.vertices
    
    with open(filepath, 'w') as f:
        # Scrivi la linea EXTERIOR
        f.write("EXTERIOR\n")
        for idx in exterior_indices:
            v = verts[idx].co
            f.write(f"{v.x} {v.y}\n")
        
        # Per ogni hole scrivi HOLE e i suoi vertici
        for hole in holes_indices:
            f.write("HOLE\n")
            for idx in hole:
                v = verts[idx].co
                f.write(f"{v.x} {v.y}\n")
        
        f.write("END\n")


### function: apply_global_shift ###
def apply_global_shift(input_path: str, output_path: str, x_offset: float, y_offset: float) -> None:
    """
    Applies a global translation to a mesh and saves the result.

    Args:
        input_path (str): Path to the input file (.ply, .obj, etc.).
        output_path (str): Path to the output file (.ply, .obj, etc.).
        x_offset (float): Translation along the X axis.
        y_offset (float): Translation along the Y axis.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    mesh = trimesh.load(input_path, force='mesh')

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"The loaded file is not a triangular mesh: {input_path}")

    shift_vector = np.array([x_offset, y_offset, 0.0])
    mesh.apply_translation(shift_vector)

    mesh.export(output_path)
    print(f"Shifted mesh saved to: {output_path}")