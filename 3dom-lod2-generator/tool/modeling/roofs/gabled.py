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
from io_utils.exporter import export_mesh_ply


### function: calculate_roof_height ###
def calculate_roof_height(base_length, slope_percent=22):
    """
    Calculates the height to apply to an edge based on a given slope percentage.

    Args:
        base_length (float): The length of the base to which the slope is applied.
        slope_percent (float): Desired slope expressed as a percentage (default is 22%).

    Returns:
        float: The height needed to achieve the desired slope.
    """
    return (slope_percent / 100.0) * base_length


### function: create_gabled_roof ###
def create_gabled_roof(base_obj, height, exterior_coords, round_edges=False):
    """
    Constructs a gabled roof on the given base mesh by creating a sloped bounding box 
    and cutting it from the extruded base. Optionally applies rounding on the outer edges.

    Steps:
    1. Clean the base mesh by merging nearby vertices.
    2. Generate a minimum-area bounding box aligned with the base.
    3. Split the bounding box along its longest side to define a ridge.
    4. Raise the ridge to create the gable shape.
    5. Align the gable box vertically to match the base extrusion height.
    6. Extrude the base mesh vertically.
    7. Perform a Boolean difference to subtract the gabled volume.
    8. Optionally round outer edges using a beveled polygon mesh.

    Parameters:
    - base_obj (bpy.types.Object): The mesh object representing the base structure.
    - height (float): Vertical height for roof extrusion.
    - exterior_coords (list of tuple): Coordinates of the outer loop for rounding.
    - round_edges (bool): Whether to apply a rounded bevel to outer edges.

    Returns:
    - bpy.types.Object: The resulting mesh object with the gabled roof.
    """
    # Clean up base mesh
    blender_ops.merge_close_vertices(base_obj)

    # Create optimal bounding box from base footprint
    bbox = blender_ops.create_optimal_bounding_box(base_obj)
    blender_ops.merge_close_vertices(bbox)
    blender_ops.limited_dissolve_all_faces(bbox)

    # Identify central edge and compute roof height
    new_edge_indices, short_edge_length = blender_ops.split_bbox_plane(bbox)
    ridge_height = calculate_roof_height(short_edge_length)

    # Form the gabled shape by raising the ridge edge
    blender_ops.move_edge_up_object(bbox, new_edge_indices, ridge_height)
    blender_ops.align_mesh_to_reference(bbox, height)
    blender_ops.move_mesh_z(bbox, -0.1)

    # Extrude the base mesh upward
    blender_ops.extrude_faces_z(base_obj, height)

    # Cut the base using the gabled volume
    blender_ops.apply_boolean_difference(base_obj, bbox, modifier_name="Boolean_Diff")

    if round_edges:
        # Create and bevel the polygon outline mesh
        round_obj = create_mesh_from_polygon("round_edge", exterior_coords, [])
        blender_ops.merge_close_vertices(round_obj)
        blender_ops.limited_dissolve_all_faces(round_obj)
        blender_ops.compute_custom_vertex_attribute(round_obj, target_coords=exterior_coords)
        blender_ops.apply_bevel_modifier(round_obj, width=2)
        blender_ops.extrude_faces_z(round_obj, height + 1)

        # Intersect the beveled outline with the roof mesh
        blender_ops.apply_boolean_intersect(base_obj, round_obj, apply=True)

        # Clean resulting geometry
        blender_ops.triangulate_mesh(base_obj)
        blender_ops.merge_close_vertices(base_obj)
        blender_ops.limited_dissolve_all_faces(base_obj)
        blender_ops.triangulate_mesh(base_obj)
    
    blender_ops.triangulate_mesh(base_obj)