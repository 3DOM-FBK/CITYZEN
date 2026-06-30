import bpy
import bmesh
from mathutils import Vector, geometry
import mathutils
import numpy as np
import os
import shutil
import math
from scipy.spatial import ConvexHull
import sys

#######################################################
# Adds the root project in the Python path
#######################################################
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
#######################################################

from modeling.min_bounding_rect import minBoundingRect


### function: clean_tmp_folder ###
def clean_tmp_folder(path="/tmp") -> bool:
    """
    Cleans a directory by removing all its contents (files and subdirectories).

    Args:
        path (str): Path to the directory to clean. Defaults to '/tmp'.

    Returns:
        bool: True if cleaning succeeded, False otherwise.
    """
    if not os.path.exists(path) or not os.path.isdir(path):
        print(f"Directory does not exist or is not a directory: {path}")
        return False

    try:
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove directory and its contents
        return True
    except Exception as e:
        print(f"Error cleaning directory {path}: {e}")
        return False


### function: clear_blender_scene ###
def clear_blender_scene():
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    data_types = [
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.armatures,
        bpy.data.objects,
        bpy.data.collections
    ]

    for data_block in data_types:
        for item in data_block:
            if item.users == 0:
                data_block.remove(item)

    # Rimuove tutte le collezioni non usate
    for collection in bpy.data.collections:
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    print("Clean Scene - Done")



### function: extrude_faces_z ###
def extrude_faces_z(obj, height):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()

    # Seleziona tutte le facce
    for f in bm.faces:
        f.select = True

    # Estrusione delle facce selezionate
    ret = bmesh.ops.extrude_face_region(bm, geom=bm.faces[:])
    
    # Prende i vertici dell'estrusione e li sposta lungo Z
    verts = [ele for ele in ret['geom'] if isinstance(ele, bmesh.types.BMVert)]
    bmesh.ops.translate(bm, verts=verts, vec=Vector((0, 0, height)))

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)


### function: get_convex_hull_2d_numpy ###
def get_convex_hull_2d_numpy(obj=None):
    """
    Computes the 2D convex hull (XY projection) of a mesh object's vertices.

    Args:
        obj (bpy.types.Object): The mesh object. If None, uses active object.

    Returns:
        np.ndarray: An (N x 2) array of 2D convex hull points in world coordinates.
    """
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        raise ValueError("No valid mesh object provided.")

    # Convert all mesh vertices to world-space and project to XY
    verts_world_xy = [ (obj.matrix_world @ v.co).to_2d() for v in obj.data.vertices ]
    verts_np = np.array([[v.x, v.y] for v in verts_world_xy])

    if len(verts_np) < 3:
        raise ValueError("Not enough vertices to compute convex hull.")

    # Compute convex hull
    hull = ConvexHull(verts_np)
    hull_coords = verts_np[hull.vertices]

    return hull_coords


### function: create_mesh_from_2d_points ###
def create_mesh_from_2d_points(points_2d, name="GeneratedMesh", z_height=0.0):
    """
    Creates a flat Blender mesh from a Nx2 numpy array of 2D points.

    Args:
        points_2d (np.ndarray): Nx2 array of (x, y) points (must be ordered for face).
        name (str): Name of the new mesh object.
        z_height (float): Z value to assign to all vertices.

    Returns:
        bpy.types.Object: The newly created mesh object.
    """
    if points_2d.shape[1] != 2:
        raise ValueError("Input must be a Nx2 NumPy array.")

    # Ensure the polygon is closed
    if not np.allclose(points_2d[0], points_2d[-1]):
        points_2d = np.vstack([points_2d, points_2d[0]])

    # Create mesh data
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    verts = [Vector((x, y, z_height)) for x, y in points_2d]
    edges = []
    faces = [list(range(len(verts)))]

    mesh.from_pydata(verts, edges, faces)
    mesh.update()

    return obj


def expand_bbox_from_center(center, rot_angle, width, height, offset=0.5):
    """
    Expands a rotated bounding box from its center, angle, width, and height.

    Args:
        center (np.ndarray): (2,) array representing the center (cx, cy).
        rot_angle (float): Rotation angle in radians.
        width (float): Original width of the bbox.
        height (float): Original height of the bbox.
        offset (float): Expansion value to apply outward.

    Returns:
        np.ndarray: (4, 2) array of 2D points representing the expanded bbox corners.
    """
    # Expand dimensions
    w = width / 2.0 + offset
    h = height / 2.0 + offset

    # Corners in local (unrotated) space
    local_corners = np.array([
        [ w, -h],
        [-w, -h],
        [-w,  h],
        [ w,  h]
    ])

    # Rotation matrix
    cos_a = math.cos(rot_angle)
    sin_a = math.sin(rot_angle)
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])

    # Rotate and translate corners
    rotated_corners = np.dot(local_corners, R.T) + center

    return rotated_corners


### function: create_optimal_bounding_box ###
def create_optimal_bounding_box(obj, name="OBB_Plane", offset=0.5):
    hull_coords = get_convex_hull_2d_numpy(obj)

    bbox = minBoundingRect(hull_coords)

    corner_points = expand_bbox_from_center(bbox[4], bbox[0], bbox[2], bbox[3], offset=offset)

    obb_obj = create_mesh_from_2d_points(corner_points, name)

    return obb_obj


### function: split_bbox_plane ###
def split_bbox_plane(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    # Calcolo lunghezze
    edge_lengths = [(e, (e.verts[0].co - e.verts[1].co).length) for e in bm.edges]
    edge_lengths.sort(key=lambda x: x[1])
    
    short_edges = [e for e, _ in edge_lengths[:2]]
    short_edge_length = edge_lengths[0][1]  # tutti e due avranno simile lunghezza

    # Deseleziona tutto, poi seleziona solo i più corti
    for e in bm.edges:
        e.select = False
    for e in short_edges:
        e.select = True

    # Suddivide e raccoglie i nuovi edge
    result = bmesh.ops.subdivide_edges(
        bm,
        edges=short_edges,
        cuts=1,
        use_grid_fill=True
    )

    new_edges = result.get('geom_inner', [])
    new_edges = [e for e in new_edges if isinstance(e, bmesh.types.BMEdge)]
    new_edge_indices = [e.index for e in new_edges]

    # Applica modifiche alla mesh
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    return new_edge_indices, short_edge_length


### function: move_edge_up_object ###
def move_edge_up_object(obj, edge_indices, height):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()

    edge = bm.edges[edge_indices[0]]

    for vert in edge.verts:
        vert.co.z += height

    bm.to_mesh(mesh)
    bm.free()
    mesh.update()


### function: align_bbox_to_reference ###
def align_mesh_to_reference(bbox_obj, height):
    def get_max_world_z(obj):
        return max((obj.matrix_world @ v.co).z for v in obj.data.vertices)

    max_z_bbox = get_max_world_z(bbox_obj)

    delta_z = height - max_z_bbox

    bbox_obj.location.z += delta_z


### function: merge_close_vertices ###
def merge_close_vertices(obj, distance=0.001):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.verts.ensure_lookup_table()

    # Merge dei vertici vicini
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)


### function: apply_boolean_difference ###
def apply_boolean_difference(obj_target, obj_cutter, modifier_name="Boolean_Diff"):
    if obj_target.type != 'MESH' or obj_cutter.type != 'MESH':
        raise TypeError("Entrambi gli oggetti devono essere mesh.")

    bpy.context.view_layer.objects.active = obj_target
    obj_target.select_set(True)
    obj_cutter.select_set(False)

    mod = obj_target.modifiers.new(name=modifier_name, type='BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.object = obj_cutter


    bpy.ops.object.modifier_apply(modifier=modifier_name)

    return obj_target


### function: apply_boolean_intersect ###
def apply_boolean_intersect(obj_a, obj_b, apply=True):
    if obj_a is None or obj_b is None:
        print("Entrambi gli oggetti devono essere specificati.")
        return

    bpy.context.view_layer.objects.active = obj_a
    bpy.ops.object.select_all(action='DESELECT')
    obj_a.select_set(True)

    mod = obj_a.modifiers.new(name="Boolean_Intersect", type='BOOLEAN')
    mod.operation = 'INTERSECT'
    mod.object = obj_b
    mod.solver = 'EXACT'

    if apply:
        bpy.ops.object.modifier_apply(modifier=mod.name)


### function: get_exterior_and_hole_loops ###
def get_exterior_and_hole_loops(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]

    vert_to_boundary_edges = {}
    for e in boundary_edges:
        for v in e.verts:
            vert_to_boundary_edges.setdefault(v.index, []).append(e)

    def extract_loop(start_edge):
        loop_verts = []
        current_edge = start_edge
        current_vert = start_edge.verts[0]
        visited = set()

        while True:
            loop_verts.append(current_vert)
            next_vert = current_edge.other_vert(current_vert)

            connected_edges = [
                e for e in vert_to_boundary_edges[next_vert.index]
                if e != current_edge and e.index not in visited
            ]
            visited.add(current_edge.index)

            if not connected_edges:
                break
            next_edge = connected_edges[0]

            current_vert = next_vert
            current_edge = next_edge

            if current_vert == loop_verts[0]:
                break

        return loop_verts

    visited_edges = set()
    loops = []

    for e in boundary_edges:
        if e.index in visited_edges:
            continue

        loop_verts = extract_loop(e)
        for i in range(len(loop_verts)):
            vertex_1 = loop_verts[i]
            vertex_2 = loop_verts[(i + 1) % len(loop_verts)]
            edge = bm.edges.get([vertex_1, vertex_2])
            if edge:
                visited_edges.add(edge.index)

        loops.append(loop_verts)

    def is_clockwise(verts):
        coords = [(v.co.x, v.co.y) for v in verts]
        area = 0
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            area += x1 * y2 - x2 * y1
        return area < 0

    def shoelace_area(verts):
        coords = [(v.co.x, v.co.y) for v in verts]
        area = 0
        for i in range(len(coords)):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % len(coords)]
            area += x1 * y2 - x2 * y1
        return area / 2

    loops_sorted = sorted(loops, key=lambda lv: abs(shoelace_area(lv)), reverse=True)
    exterior_loop = loops_sorted[0]
    holes = loops_sorted[1:]

    # CGAL: esterno antiorario
    if is_clockwise(exterior_loop):
        exterior_loop.reverse()

    # CGAL: fori orari
    for hole_loop in holes:
        if not is_clockwise(hole_loop):
            hole_loop.reverse()

    exterior_indices = [v.index for v in exterior_loop]
    holes_indices = [[v.index for v in hole] for hole in holes]

    bm.free()
    return exterior_indices, holes_indices


### function: delete_downward_faces ###
def delete_downward_faces(obj=None):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("No object selected.")
        return

    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    bm.normal_update()

    faces_to_delete = [f for f in bm.faces if f.normal.z < 0]

    bmesh.ops.delete(bm, geom=faces_to_delete, context='FACES')

    verts_to_delete = [v for v in bm.verts if len(v.link_faces) == 0]
    bmesh.ops.delete(bm, geom=verts_to_delete, context='VERTS')

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()


### function: delete_facing_up_faces ###
def delete_facing_up_faces(obj, threshold=0.0):
    if obj.type != 'MESH':
        print("Selected object is not a mesh")
        return

    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data

    bm = bmesh.new()
    bm.from_mesh(mesh)

    bm.faces.ensure_lookup_table()

    up_faces = [f for f in bm.faces if f.normal.z > threshold]

    bmesh.ops.delete(bm, geom=up_faces, context='FACES')

    bm.to_mesh(mesh)
    bm.free()


### function: get_mesh_height ###
def get_mesh_height(obj=None):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("No valid mesh selected")
        return None

    bpy.ops.object.mode_set(mode='OBJECT')

    zs = [obj.matrix_world @ v.co for v in obj.data.vertices]
    z_values = [v.z for v in zs]

    z_min = min(z_values)
    z_max = max(z_values)
    height = z_max - z_min

    return height


### function: align_bottom_to_top ###
def align_bottom_to_top(source_obj, reference_obj):
    if not source_obj or not reference_obj:
        print("No valid Objects.")
        return

    source_zs = [source_obj.matrix_world @ v.co for v in source_obj.data.vertices]
    reference_zs = [reference_obj.matrix_world @ v.co for v in reference_obj.data.vertices]

    source_z_min = min(v.z for v in source_zs)
    reference_z_max = max(v.z for v in reference_zs)

    delta_z = reference_z_max - source_z_min

    source_obj.location.z += delta_z


### function: join_meshes ###
def join_meshes(obj1, obj2):
    if obj1.type != 'MESH' or obj2.type != 'MESH':
        print("Both objects must be of type MESH.")
        return

    bpy.ops.object.select_all(action='DESELECT')
    obj1.select_set(True)
    obj2.select_set(True)
    bpy.context.view_layer.objects.active = obj1

    bpy.ops.object.join()


### function: bevel_vertical_edges ###
def bevel_vertical_edges(obj=None, angle_threshold_deg=10, width=0.03, segments=3, profile=0.5):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("No object selected")
        return False

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    for e in bm.edges:
        e.select = False

    z_axis = Vector((0, 0, 1))
    angle_thresh_rad = math.radians(angle_threshold_deg)

    for e in bm.edges:
        vec = (e.verts[1].co - e.verts[0].co).normalized()
        angle = vec.angle(z_axis)
        if angle < angle_thresh_rad or abs(angle - math.pi) < angle_thresh_rad:
            e.select = True

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='EDGE')
    bpy.ops.mesh.bevel(offset=width, segments=segments, profile=profile, affect='EDGES')
    bpy.ops.object.mode_set(mode='OBJECT')
    return True


### function: bevel_vertical_edges ###
def limited_dissolve_all_faces(obj=None, angle_limit=0.01):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("Nessun oggetto mesh attivo o non è una mesh.")
        return

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.dissolve_limited(angle_limit=angle_limit)
    bpy.ops.object.mode_set(mode='OBJECT')


### function: compute_custom_vertex_attribute ###
def compute_custom_vertex_attribute(obj=None, attr_name="bevel_weight_vert", default_value=1.0, target_coords=[]):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("Oggetto non valido.")
        return

    mesh = obj.data
    target_coords = [Vector(c) for c in target_coords]

    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])

    attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')

    bpy.ops.object.mode_set(mode='OBJECT')
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    epsilon = 1e-6
    values = [0.0] * len(bm.verts)

    for i, v in enumerate(bm.verts):
        if any((v.co - c).length < epsilon for c in target_coords):
            min_dist = None
            for edge in v.link_edges:
                other = edge.other_vert(v)
                # Calcola distanza solo se anche il vertice collegato è nella lista target
                if any((other.co - c).length < epsilon for c in target_coords):
                    dist = (v.co - other.co).length
                    if min_dist is None or dist < min_dist:
                        min_dist = dist

            if min_dist is None:
                val = 0.0
            elif default_value < (min_dist / 2.0):
                val = 1.0
            else:
                val = ((min_dist / 2.0) / default_value) - 0.05

            values[i] = max(0.0, val)
        else:
            values[i] = 0.0

    bm.free()

    for i, v in enumerate(values):
        attr.data[i].value = v

    mesh.update()


### function: apply_bevel_modifier ###
def apply_bevel_modifier(obj=None, name="Bevel_Weight", width=1, segments=4):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("Oggetto non valido.")
        return

    mod = obj.modifiers.new(name=name, type='BEVEL')
    mod.limit_method = 'WEIGHT'
    mod.width = width
    mod.segments = segments
    mod.profile = 0.5
    mod.use_clamp_overlap = True
    mod.affect = 'VERTICES'

    bpy.ops.object.modifier_apply(modifier=mod.name)


### function: triangulate_mesh ###
def triangulate_mesh(obj=None):
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("No object selected")
        return

    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)

    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()


### function: collapse_top_vertices_to_center ###
def collapse_top_vertices_to_center(obj=None):
    """
    Collapses all non-base vertices of a mesh to their center point in Z,
    leaving the base of the object (lowest Z vertices) untouched.

    Args:
        obj (Object): Blender object to operate on. Defaults to active object.
    """
    if obj is None:
        obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        print("No valid mesh object selected.")
        return

    bpy.ops.object.mode_set(mode='OBJECT')
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    # Find min Z
    min_z = min(v.co.z for v in bm.verts)

    # Find all vertices to collapse
    top_verts = [v for v in bm.verts if v.co.z > min_z]

    if not top_verts:
        print("No non-base vertices found.")
        bm.free()
        return

    # Compute center position
    center = sum((v.co for v in top_verts), Vector()) / len(top_verts)

    # modify vertices position
    for v in top_verts:
        v.co = center

    bm.to_mesh(mesh)
    mesh.update()
    bm.free()
    print("Top vertices collapsed to center.")



### function: align_top_vertex_to_plane ###
def align_top_vertex_to_plane(obj=None):
    """
    Aligns the highest vertex of each selected triangular face to a vertical plane
    defined by the two lower vertices of the same face.

    The function operates in Edit Mode on the given mesh object (or the active object if none is provided).
    For each selected triangle:
    - Identifies the top vertex (with highest Z coordinate).
    - Constructs a vertical plane (Z axis up) through the other two base vertices.
    - Projects the top vertex onto this plane along the direction of the external edge connected to it.

    This is useful for flattening or shaping geometry such as sloped roof surfaces.

    Args:
        obj (bpy.types.Object, optional): The target mesh object. Defaults to the active object.
    """
    if obj is None:
        obj = bpy.context.active_object
    if obj is None or obj.type != 'MESH':
        print("No valid mesh object selected.")
        return

    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    for f in bm.faces:
        if not f.select or len(f.verts) != 3:
            continue

        verts = sorted(f.verts, key=lambda v: v.co.z, reverse=True)
        top_v = verts[0]
        base_vertex_1, base_vertex_2 = verts[1], verts[2]

        base_dir = (base_vertex_2.co - base_vertex_1.co).normalized()
        
        up = Vector((0, 0, 1))
        normal = base_dir.cross(up).normalized()

        plane_point = base_vertex_1.co

        external_edge = None
        for e in top_v.link_edges:
            if f not in e.link_faces:
                external_edge = e
                break

        if external_edge:
            vertex_1, vertex_2 = external_edge.verts
            other_vert = vertex_1 if vertex_1 != top_v else vertex_2

            intersection = mathutils.geometry.intersect_line_plane(
                vertex_1.co, vertex_2.co,
                plane_point, normal,
                False
            )

            if intersection:
                top_v.co = intersection
        else:
            print("--> No projection_dir found.")

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')


### function: move_mesh_z ###
def move_mesh_z(obj=None, delta_z=0.0):
    """
    Moves the given mesh object along the Z-axis by the specified amount.

    Args:
        obj (bpy.types.Object): The Blender object to move. If None, uses the active object.
        delta_z (float): Distance to move along the Z-axis.
    """
    if obj is None:
        obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("No valid mesh object selected.")
        return

    obj.location.z += delta_z


### function: move_mesh_z ###
def flatten_mesh_to_z(obj, z_min):
    for vert in obj.data.vertices:
        vert.co.z = z_min


### function: count_mesh_points ###
def count_mesh_points(obj):
    if obj and obj.type == 'MESH':
        num_points = len(obj.data.vertices)
        print(f"Numero di punti (vertici) nella mesh '{obj.name}': {num_points}")
        return num_points
    else:
        print(f"L'oggetto '{obj_name}' non è una mesh valida.")
        return -1


### function: duplicate_object ###
def duplicate_object(obj, new_name):
    obj_copy = obj.copy()
    obj_copy.data = obj.data.copy()
    obj_copy.name = new_name
    bpy.context.collection.objects.link(obj_copy)
    return obj_copy
