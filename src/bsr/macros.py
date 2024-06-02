__all__ = ["clear_mesh_objects"]

import bpy


def clear_mesh_objects() -> None:
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()
