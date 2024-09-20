__doc__ = """
Pose class for creating and updating poses in Blender
"""
__all__ = ["Pose"]

import bpy
from numpy.typing import NDArray

from bsr.geometry.primitives.simple import Cylinder, Sphere
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

        self._build(positions, directors)

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """
        Return the dictionary of Blender objects: spheres and cylinders
        """
        return self._bpy_objs

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

            sphere = Sphere(
                tip_position,
                self.__unit_length * self.__ratio,
            )
            self.spheres.append(sphere)

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

    def set_keyframe(self, keyframe: int) -> None:
        """
        Set the keyframe for the pose object
        """
        for shperes in self.spheres:
            shperes.set_keyframe(keyframe)

        for cylinder in self.cylinders:
            cylinder.set_keyframe(keyframe)
