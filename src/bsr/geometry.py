__doc__ = """
This module provides a set of geometry-mesh interfaces for blender objects.
"""
__all__ = ["BlenderMeshInterfaceProtocol", "Sphere", "Cylinder"]

from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    Protocol,
    Type,
    TypedDict,
    TypeVar,
)
from typing_extensions import Self

import colorsys

import bpy
import numpy as np

MeshDataType = dict[str, Any]

S = TypeVar("S", bound="BlenderMeshInterfaceProtocol")
P = ParamSpec("P")


class BlenderMeshInterfaceProtocol(Protocol):
    """
    This protocol defines the interface for Blender mesh objects.
    """

    @property
    def states(self) -> MeshDataType:
        """Returns the current state of the mesh object."""

    # TODO: For future implementation
    # @property
    # def data(self): ...

    @property
    def object(self) -> bpy.types.Object:
        """Returns associated Blender object."""

    @classmethod
    def create(cls: Type[S], states: MeshDataType) -> S:
        """Creates a new mesh object with the given states."""

    def update_states(self, *args: Any) -> bpy.types.Object:
        """Updates the mesh object with the given states."""

    # def update_material(self, material) -> None: ...  # TODO: For future implementation


class Sphere:
    """
    This class provides a mesh interface for Blender Sphere objects.
    Sphere objects are created with the given position and radius.

    Parameters
    ----------
    position : np.ndarray
        The position of the sphere object.
    radius : float
        The radius of the sphere object.
    """

    def __init__(self, position: np.ndarray, radius: float) -> None:
        self._obj = self._create_sphere(position, radius)

    @classmethod
    def create(cls, states: MeshDataType) -> "Sphere":
        return cls(states["position"], states["radius"])

    @property
    def object(self) -> bpy.types.Object:
        return self._obj

    @property
    def states(self) -> MeshDataType:
        states = {
            "position": np.array(
                [
                    self.object.location.x,
                    self.object.location.y,
                    self.object.location.z,
                ]
            ),
            "radius": self.object.radius,
        }
        return states

    def update_states(
        self, position: np.ndarray | None = None, radius: float | None = None
    ) -> bpy.types.Object:
        if position is not None:
            self.object.location.x = position[0]
            self.object.location.y = position[1]
            self.object.location.z = position[2]
        if radius is not None:
            self.object.radius = radius
        return self.object

    def _create_sphere(
        self, position: np.ndarray, radius: float
    ) -> bpy.types.Object:
        """
        Creates a new sphere object with the given position and radius.
        """
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=position)
        return bpy.context.active_object


# FIXME: This class needs to be modified to conform to the BlenderMeshInterfaceProtocol
class Cylinder:
    """
    TODO: Add documentation
    """

    def __init__(self, pos1, pos2):
        self.obj = self.create_cylinder(pos1, pos2)
        self.mat = bpy.data.materials.new(name="cyl_mat")
        self.obj.active_material = self.mat

    def create_cylinder(self, pos1, pos2):
        depth, center, angles = self.calc_cyl_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.005, depth=1, location=center
        )
        cylinder = bpy.context.active_object
        cylinder.rotation_euler = (0, angles[1], angles[0])
        cylinder.scale[2] = depth
        return cylinder

    def calc_cyl_orientation(self, pos1, pos2):
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
        depth, center, angles = self.calc_cyl_orientation(pos1, pos2)
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


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {"position": np.array([0, 0, 0]), "radius": 1.0}
    _: BlenderMeshInterfaceProtocol = Sphere.create(data)
    data = {
        "position_1": np.array([0, 0, 0]),
        "position_2": np.array([1, 1, 1]),
        "radius": 1.0,
    }
    _: BlenderMeshInterfaceProtocol = Cylinder.create(data)
