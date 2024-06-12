__doc__ = """
This module provides a set of geometry-mesh interfaces for blender objects.
"""
__all__ = ["Sphere", "Cylinder"]

from typing import TYPE_CHECKING

import bpy
import numpy as np
from numpy.typing import NDArray

from .mixin import KeyFrameControlMixin
from .protocol import BlenderMeshInterfaceProtocol, MeshDataType


# TODO: use numba
def calculate_cylinder_orientation(position_1, position_2):
    depth = np.linalg.norm(position_2 - position_1)
    dz = position_2[2] - position_1[2]
    dy = position_2[1] - position_1[1]
    dx = position_2[0] - position_1[0]
    center = (position_1 + position_2) / 2
    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz / depth)
    angles = np.array([phi, theta])
    return depth, center, angles


class Sphere(KeyFrameControlMixin):
    """
    This class provides a mesh interface for Blender Sphere objects.
    Sphere objects are created with the given position and radius.

    Parameters
    ----------
    position : NDArray
        The position of the sphere object. (3D)
    radius : float
        The radius of the sphere object.

    """

    def __init__(self, position: NDArray, radius: float) -> None:
        self._obj = self._create_sphere()
        self.update_states(position, radius)

    @classmethod
    def create(cls, states: MeshDataType) -> "Sphere":
        """
        Basic factory method to create a new Sphere object.
        """
        return cls(states["position"], states["radius"])

    @property
    def object(self) -> bpy.types.Object:
        """
        Access the Blender object.
        """
        return self._obj

    def update_states(self, position: NDArray, radius: float) -> None:
        """
        Updates the position and radius of the sphere object.

        Parameters
        ----------
        position : NDArray
            The new position of the sphere object.
        radius : float
            The new radius of the sphere object.
        """
        if position is not None:
            self.object.location.x = position[0]
            self.object.location.y = position[1]
            self.object.location.z = position[2]
        if radius is not None:
            self.object.scale = (radius, radius, radius)

    def _create_sphere(self) -> bpy.types.Object:
        """
        Creates a new sphere object with the given position and radius.
        """
        bpy.ops.mesh.primitive_uv_sphere_add()
        return bpy.context.active_object

    def set_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self.object.keyframe_insert(data_path="location", frame=keyframe)


# FIXME: This class needs to be modified to conform to the BlenderMeshInterfaceProtocol
class Cylinder(KeyFrameControlMixin):
    """
    TODO: Add documentation
    """

    def __init__(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius: float,
    ):
        self._obj = self._create_cylinder(
            position_1,
            position_2,
            radius,
        )

    @classmethod
    def create(cls, states: MeshDataType) -> "Cylinder":
        return cls(states["position_1"], states["position_2"], states["radius"])

    @property
    def object(self) -> bpy.types.Object:
        return self._obj

    def update_states(self, position_1, position_2, radius):
        depth, center, angles = calculate_cylinder_orientation(
            position_1, position_2
        )
        self.object.location = center
        self.object.rotation_euler = (0, angles[1], angles[0])
        self.object.scale[2] = depth
        self.object.scale[0] = radius
        self.object.scale[1] = radius

    def _create_cylinder(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius: float,
    ) -> bpy.types.Object:
        """
        Creates a new cylinder object with the given end positions, radius, centerpoint and depth.
        """
        depth, center, angles = calculate_cylinder_orientation(
            position_1, position_2
        )
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=1.0)
        cylinder = bpy.context.active_object
        cylinder.rotation_euler = (0, angles[1], angles[0])
        cylinder.scale[2] = depth
        return cylinder

    def set_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self.object.keyframe_insert(data_path="location", frame=keyframe)
        self.object.keyframe_insert(data_path="rotation_euler", frame=keyframe)
        self.object.keyframe_insert(data_path="scale", frame=keyframe)
        # self.object.keyframe_insert(data_path="diffuse_color", frame=keyframe)

    # def update_color(self, val):
    #    # computing deformation heat-map
    #    max_def = 0.07

    #    h = -np.sqrt(val) / max_def + 240 / 360
    #    v = np.sqrt(val) / max_def * 0.5 + 0.5

    #    r, g, b = colorsys.hsv_to_rgb(h, 1, v)
    #    self.mat.diffuse_color = (r, g, b, a)


# TODO: Will be implemented in the future
class Frustum(KeyFrameControlMixin):  # pragma: no cover
    """
    This class provides a mesh interface for Blender Frustum objects.
    Frustum objects are created with the given positions and radii.

    Parameters
    ----------
    position_1 : NDArray
        The position of the first end of the frustum object. (3D)
    position_2 : NDArray
        The position of the second end of the frustum object. (3D)
    radius_1 : float
        The radius of the first end of the frustum object.
    radius_2 : float
        The radius of the second end of the frustum object.
    """

    def __init__(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius_1: float,
        radius_2: float,
    ):
        raise NotImplementedError
        # self._obj = self._create_frustum(
        #    position_1, position_2, radius_1, radius_2
        # )
        # self.update_states(position_1, position_2, radius_1, radius_2)

        # self.mat = bpy.data.materials.new(name="cyl_mat")
        # self.obj.active_material = self.mat

    @classmethod
    def create(cls, states: MeshDataType) -> "Frustum":
        raise NotImplementedError
        # return cls(
        #    states["position_1"],
        #    states["position_2"],
        #    states["radius_1"],
        #    states["radius_2"],
        # )

    @property
    def object(self) -> bpy.types.Object:
        raise NotImplementedError
        return self._obj

    def _create_frustum(self, position_1, position_2, radius_1, radius_2):
        raise NotImplementedError
        # depth, center, angles = calculate_cylinder_orientation(
        #     position_1, position_2
        # )
        # bpy.ops.mesh.primitive_cone_add(
        #     radius1=radius_1, radius2=radius_2, depth=1,
        # )
        # frustum = bpy.context.active_object
        # frustum.rotation_euler = (0, angles[1], angles[0])
        # frustum.location = center
        # frustum.scale[2] = depth
        # return frustum

    def update_states(self, position_1, position_2, radius_1, radius_2):
        raise NotImplementedError

    def set_keyframe(self, keyframe: int) -> None:
        raise NotImplementedError


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {"position": np.array([0, 0, 0]), "radius": 1.0}
    _: BlenderMeshInterfaceProtocol = Sphere.create(data)
    data = {
        "position_1": np.array([0, 0, 0]),
        "position_2": np.array([1, 1, 1]),
        "radius": 1.0,
    }
    _: BlenderMeshInterfaceProtocol = Cylinder.create(data)  # type: ignore[no-redef]
    data = {
        "position_1": np.array([0, 0, 0]),
        "position_2": np.array([1, 1, 1]),
        "radius_1": 1.0,
        "radius_2": 1.5,
    }
    _: BlenderMeshInterfaceProtocol = Frustum.create(data)  # type: ignore[no-redef]
