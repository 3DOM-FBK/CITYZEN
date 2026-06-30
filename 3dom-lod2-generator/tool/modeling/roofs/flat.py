import sys
import os

#######################################################
# Adds the root project in the Python path
#######################################################
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#######################################################

import blender_ops as blender_ops
from shapefile.converter import create_mesh_from_polygon


### function: create_flat_roof ###
def create_flat_roof(base_obj, height, exterior_coords, round_edges=False):
    """
    Creates a flat roof by extruding the base object upward. Optionally applies 
    edge rounding on the outer perimeter using a bevel modifier.

    Parameters:
    - base_obj (Object): The Blender mesh object representing the building base.
    - height (float): The extrusion height to form the flat roof.
    - exterior_coords (list of tuple): Coordinates of the outer loop for edge rounding.
    - round_edges (bool): If True, apply rounding to the outer edges using a beveled mesh.

    Returns:
    - Object: The final modified mesh object with a flat roof.
    """

    # Prepare base mesh
    blender_ops.merge_close_vertices(base_obj)
    blender_ops.extrude_faces_z(base_obj, height)

    if round_edges:
        # Create beveled outline mesh
        round_obj = create_mesh_from_polygon("round_edge", exterior_coords, [])
        blender_ops.merge_close_vertices(round_obj)
        blender_ops.limited_dissolve_all_faces(round_obj)
        blender_ops.compute_custom_vertex_attribute(round_obj, target_coords=exterior_coords)
        blender_ops.apply_bevel_modifier(round_obj, width=2)
        blender_ops.extrude_faces_z(round_obj, height + 1)

        # Apply boolean intersection to round the base object's edges
        blender_ops.apply_boolean_intersect(base_obj, round_obj, apply=True)

        # Clean the resulting mesh
        blender_ops.triangulate_mesh(base_obj)
        blender_ops.merge_close_vertices(base_obj)
        blender_ops.limited_dissolve_all_faces(base_obj)
        blender_ops.triangulate_mesh(base_obj)

    blender_ops.triangulate_mesh(base_obj)