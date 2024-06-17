from numbers import Number

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.geometry import BlenderMeshInterfaceProtocol
from bsr.macros import scene_update


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


def validate_position(position: NDArray) -> None:  # pragma: no cover
    if position.shape != (3,):
        raise ValueError("The shape of the position is incorrect.")
    if np.isnan(position).any():
        raise ValueError("The position contains NaN values.")


def validate_radius(radius: float) -> None:  # pragma: no cover
    if not isinstance(radius, Number) or radius <= 0:
        raise ValueError("The radius must be a positive float.")
    if np.isnan(radius):
        raise ValueError("The radius contains NaN values.")
