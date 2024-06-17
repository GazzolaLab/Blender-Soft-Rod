# from numbers import Number

import bpy
import numpy as np

from bsr.geometry import BlenderMeshInterfaceProtocol
from bsr.macros import scene_update

# from numpy.typing import NDArray


def get_mesh_limit(interface: BlenderMeshInterfaceProtocol):
    """(For testing) Given blender mesh object, return xyz limit"""

    obj = interface.object
    scene_update()

    vertices_coords = []
    for v in obj.data.vertices:
        global_coord = obj.matrix_world @ v.co
        vertices_coords.append(list(global_coord))
    vertices_coords = np.array(vertices_coords)

    x_min, y_min, z_min = np.min(vertices_coords, axis=0)
    x_max, y_max, z_max = np.max(vertices_coords, axis=0)
    return x_min, x_max, y_min, y_max, z_min, z_max
