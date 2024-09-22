__doc__ = """
Pose class for creating and updating poses in Blender
"""
__all__ = ["Pose"]

from typing import TYPE_CHECKING, Any

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.geometry.primitives.simple import Cylinder, Sphere
from bsr.geometry.protocol import CompositeProtocol
from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class Pose(KeyFrameControlMixin):
    """
    Pose class for managing visualization and rendering in Blender

    Parameters
    ----------
    positions : NDArray
        The positions of pose. Expected shape is (n_dim,).
        n_dim = 3
    directors : NDArray
        The directors of the pose. Expected shape is (n_dim, n_dim).
        n_dim = 3

    """

    input_states = {"positions", "directors"}

    def __init__(
        self,
        positions: NDArray,
        directors: NDArray,
        unit_length: float = 1.0,
        thickness_ratio: float = 0.1,
    ) -> None:
        # create sphere and cylinder objects
        self.spheres: list[Sphere] = []
        self.cylinders: list[Cylinder] = []
        self._bpy_objs: dict[str, bpy.types.Object] = {
            "spheres": self.spheres,
            "cylinders": self.cylinders,
        }
        self.__unit_length = unit_length
        self.__ratio = thickness_ratio

        # create sphere and cylinder materials
        self.spheres_material: list[bpy.types.Material] = []
        self.cylinders_material: list[bpy.types.Material] = []
        self._bpy_materials: dict[str, bpy.types.Material] = {
            "spheres": self.spheres_material,
            "cylinders": self.cylinders_material,
        }

        self._build(positions, directors)

    @property
    def material(self) -> dict[str, bpy.types.Material]:
        """
        Return the dictionary of Blender materials: spheres and cylinders
        """
        return self._bpy_materials

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """
        Return the dictionary of Blender objects: spheres and cylinders
        """
        return self._bpy_objs

    @classmethod
    def create(cls, states: dict[str, NDArray]) -> "Pose":
        """
        Create a Pose object from the given states

        States must have the following keys: position(n_dim,), directors(n_dim, n_dim)
        """
        positions = states["positions"]
        directors = states["directors"]
        return cls(positions, directors)

    def _build(self, positions: NDArray, directors: NDArray) -> None:
        """
        Build the pose object from the given positions and directors
        """
        # create the sphere object at the positions
        sphere = Sphere(
            positions,
            self.__unit_length * self.__ratio,
        )
        self.spheres.append(sphere)

        # create cylinder and sphere objects for each director
        for i in range(directors.shape[1]):
            tip_position = positions + directors[:, i] * self.__unit_length
            cylinder = Cylinder(
                positions,
                tip_position,
                self.__unit_length * self.__ratio,
            )
            self.cylinders.append(cylinder)
            self.cylinders_material.append(cylinder.material)

            sphere = Sphere(
                tip_position,
                self.__unit_length * self.__ratio,
            )
            self.spheres.append(sphere)
            self.spheres_material.append(sphere.material)

    def update_states(self, positions: NDArray, directors: NDArray) -> None:
        """
        Update the states of the pose object
        """
        self.spheres[0].update_states(positions)

        for i, cylinder in enumerate(self.cylinders):
            tip_position = positions + directors[:, i] * self.__unit_length
            cylinder.update_states(positions, tip_position)

            sphere = self.spheres[i + 1]
            sphere.update_states(tip_position)

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the pose object
        """
        for shperes in self.spheres:
            shperes.update_material(**kwargs)

        for cylinder in self.cylinders:
            cylinder.update_material(**kwargs)

    def set_keyframe(self, keyframe: int) -> None:
        """
        Set the keyframe for the pose object
        """
        for shperes in self.spheres:
            shperes.set_keyframe(keyframe)

        for cylinder in self.cylinders:
            cylinder.set_keyframe(keyframe)


if TYPE_CHECKING:
    data = {
        "position": np.array([0.0, 0.0, 0.0]),
        "directors": np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    }
    _: CompositeProtocol = Pose.create(data)
