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

        self._build(position, directors)

    @property
    def object(self) -> dict[str, bpy.types.Object]:
        """
        Return the dictionary of Blender objects: spheres and cylinders
        """
        return self._bpy_objs

    def _build(self, position: NDArray, directors: NDArray) -> None:
        """
        Build the pose object from the given position and directors
        """
        # create the sphere object at the position
        sphere = Sphere(
            position,
            self.__unit_length * self.__ratio,
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

    def update_material(self, **kwargs) -> None:
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
