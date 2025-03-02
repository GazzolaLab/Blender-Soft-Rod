import bpy
import numpy as np

from bsr.blender_commands.macros import scene_update
from bsr.geometry.protocol import BlenderMeshInterfaceProtocol


def get_mesh_limit(interface: BlenderMeshInterfaceProtocol):
    """(For testing) Given blender mesh object, return xyz limit"""
    obj = interface.object

    if obj.type == "CURVE":
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.ops.object.convert(target="MESH")

    scene_update()

    vertices_coords = []
    for v in obj.data.vertices:
        global_coord = obj.matrix_world @ v.co
        vertices_coords.append(list(global_coord))
    vertices_coords = np.array(vertices_coords)

    x_min, y_min, z_min = np.min(vertices_coords, axis=0)
    x_max, y_max, z_max = np.max(vertices_coords, axis=0)
    return x_min, x_max, y_min, y_max, z_min, z_max
