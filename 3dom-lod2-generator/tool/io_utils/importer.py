import bpy
import os


def import_ply(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    bpy.ops.object.select_all(action='DESELECT')

    bpy.ops.wm.ply_import(filepath=filepath)

    imported_obj = bpy.context.selected_objects[0]
    return imported_obj

