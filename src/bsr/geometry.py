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
        depth, center, angles = self.calc_cyl_orientation(
            position_1, position_2
        )
        self.object.location = center
        self.object.rotation_euler = (0, angles[1], angles[0])
        self.object.scale[2] = depth
        self.object.scale[0] = radius
        self.object.scale[1] = radius

    def calc_cyl_orientation(self, position_1, position_2):
        position_1 = np.array(position_1)
        position_2 = np.array(position_2)
        depth = np.linalg.norm(position_2 - position_1)
        dz = position_2[2] - position_1[2]
        dy = position_2[1] - position_1[1]
        dx = position_2[0] - position_1[0]
        center = (position_1 + position_2) / 2
        phi = np.arctan2(dy, dx)
        theta = np.arccos(dz / depth)
        angles = np.array([phi, theta])
        return depth, center, angles

    def _create_cylinder(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius: float,
    ) -> bpy.types.Object:
        """
        Creates a new cylinder object with the given end positions, radius, centerpoint and depth.
        """
        depth, center, angles = self.calc_cyl_orientation(
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

    # def update_color(self, r, g, b, a):
    #    self.mat.diffuse_color = (r, g, b, a)


# TODO: Refactor
class Frustum(KeyFrameControlMixin):
    def __init__(self, pos1, pos2, radius1, radius2):
        self.obj = self.create_frustum(pos1, pos2, radius1, radius2)
        self.mat = bpy.data.materials.new(name="cyl_mat")
        self.obj.active_material = self.mat

    def create_frustum(self, pos1, pos2, radius1, radius2):
        depth, center, angles = self.calc_frust_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cone_add(
            radius1=radius1, radius2=radius2, depth=1, location=center
        )
        frustum = bpy.context.active_object
        frustum.rotation_euler = (0, angles[1], angles[0])
        frustum.scale[2] = depth
        return frustum

    def calc_frust_orientation(self, pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        depth = np.linalg.norm(pos2 - pos1)
        dz = pos2[2] - pos1[2]
        dy = pos2[1] - pos1[1]
        dx = pos2[0] - pos1[0]
        center = (pos1 + pos2) / 2
        phi = np.arctan2(dy, dx)
        theta = np.arccos(dz / depth)
        angles = np.array([phi, theta])
        return depth, center, angles

    def update_position(self, pos1, pos2):
        depth, center, angles = self.calc_frust_orientation(pos1, pos2)
        self.obj.location = (center[0], center[1], center[2])
        self.obj.rotation_euler = (0, angles[1], angles[0])
        self.obj.scale[2] = depth

        # computing deformation heat-map
        max_def = 0.07

        h = (
            -np.sqrt(self.obj.location[0] ** 2 + self.obj.location[2] ** 2)
            / max_def
            + 240 / 360
        )
        v = (
            np.sqrt(self.obj.location[0] ** 2 + self.obj.location[2] ** 2)
            / max_def
            * 0.5
            + 0.5
        )

        r, g, b = colorsys.hsv_to_rgb(h, 1, v)
        self.update_color(r, g, b, 1)

    def update_color(self, r, g, b, a):
        self.mat.diffuse_color = (r, g, b, a)

    def update(self, pos1, pos2, time_step):
        self.update_position(pos1, pos2)
        # adding to keyframe
        self.obj.keyframe_insert(data_path="location", frame=time_step)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=time_step)
        self.obj.keyframe_insert(data_path="scale", frame=time_step)
        self.mat.keyframe_insert(data_path="diffuse_color", frame=time_step)


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
