__doc__ = """
Pose class for creating and updating poses in Blender
"""
__all__ = ["Pose"]

from typing import TYPE_CHECKING

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
    position : NDArray
        The position of pose. Expected shape is (n_dim,).
        n_dim = 3
    directors : NDArray
        The directors of the pose. Expected shape is (n_dim, n_dim).
        n_dim = 3

    """

    input_states = {"position", "directors"}

    def __init__(
        self,
        position: NDArray,
        directors: NDArray,
    ) -> None:
        # create sphere and cylinder objects
        self.spheres: list[Sphere] = []
        self.cylinders: list[Cylinder] = []
        self._bpy_objs: dict[str, bpy.types.Object] = {
            "spheres": self.spheres,
            "cylinders": self.cylinders,
        }
        self.__unit_length = 1.0
        self.__ratio = 0.1

        self._build(position, directors)

    def set_unit_length(self, unit_length: float) -> None:
        """
        Set the unit length of the pose object
        """
        self.__unit_length = unit_length

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

        States must have the following keys: position(n_dim,), director(n_dim, n_dim), unit_length (optional)
        """
        pose = cls(**states)
        return pose

    def _build(self, position: NDArray, directors: NDArray) -> None:
        """
        Build the pose object from the given position and directors
        """
        # create the sphere object at the position
        sphere = Sphere(
            position,
            3 * self.__unit_length * self.__ratio,
        )
        self.spheres.append(sphere)

        # create cylinder and sphere objects for each director
        for i in range(directors.shape[1]):
            tip_position = position + directors[:, i] * self.__unit_length
            cylinder = Cylinder(
                position,
                tip_position,
                self.__unit_length * self.__ratio,
            )
            self.cylinders.append(cylinder)

            sphere = Sphere(
                tip_position,
                self.__unit_length * self.__ratio,
            )
            self.spheres.append(sphere)

    def update_states(self, position: NDArray, directors: NDArray) -> None:
        """
        Update the states of the pose object
        """
        self.spheres[0].update_states(position)

        for i, cylinder in enumerate(self.cylinders):
            tip_position = position + directors[:, i] * self.__unit_length
            cylinder.update_states(position, tip_position)

            sphere = self.spheres[i + 1]
            sphere.update_states(tip_position)

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
        "directors": np.diag([1.0, 1.0, 1.0]),
    }
    _: CompositeProtocol = Pose.create(data)
