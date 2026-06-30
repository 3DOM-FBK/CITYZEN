import sys
import bpy
import bmesh
from shapely.geometry import Polygon, Point
from shapely.ops import triangulate


### function: create_mesh_from_polygon ###
def create_mesh_from_polygon(name, exterior, holes):
    poly = Polygon(shell=exterior, holes=holes)
    tris = triangulate(poly)

    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    bm = bmesh.new()
    faces = []

    for tri in tris:
        coords = list(tri.exterior.coords)[:-1]
        verts = [bm.verts.new((x, y, z)) for x, y, z in coords]
        face = bm.faces.new(verts)
        faces.append(face)

    bm.faces.ensure_lookup_table()

    full_polygon = Polygon(shell=exterior, holes=holes)

    to_remove = []
    for face in bm.faces:
        verts = face.verts
        x = sum(v.co.x for v in verts) / len(verts)
        y = sum(v.co.y for v in verts) / len(verts)
        pt = Point(x, y)

        if not full_polygon.contains(pt):
            to_remove.append(face)

    bmesh.ops.delete(bm, geom=to_remove, context='FACES')

    bm.to_mesh(mesh)
    bm.free()

    return obj
