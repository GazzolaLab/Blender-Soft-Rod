__all__ = ["clear_mesh_objects"]

import bpy


def clear_mesh_objects() -> None:
    # Clear existing mesh objects in the scene
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="MESH")
    bpy.ops.object.delete()

def scene_update() -> None:
    """
    Update the scene

    Used to update object's matrix_world after transformations
    (https://blender.stackexchange.com/questions/27667/incorrect-matrix-world-after-transformation)
    """
    bpy.context.view_layer.update()

def clear_materials() -> None:
    # Clear existing materials in the scene
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
