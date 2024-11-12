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
    This class provides a mesh interface for Pose objects.
    Pose objects are created using given positions and directors.


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
        """
        Pose class constructor
        """
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
        Basic factory method to create a new Pose object.
        States must have the following keys: positions(n_dim,), directors(n_dim, n_dim)

        Parameters
        ----------
        states: dict[str, NDArray]
            A dictionary where keys are state names and values are NDArrays.

        Returns
        -------
        Pose
            An object of Pose class containing the predefined states
        """
        positions = states["positions"]
        directors = states["directors"]
        pose = cls(positions, directors)
        return pose

    def _build(self, positions: NDArray, directors: NDArray) -> None:
        """
        Populates the positions and directors of the Spheres and Cylinders into a Pose Object

        Parameters
        ----------
        positions: NDArray
            An array of shape (n_dim,) that stores the positions of the Pose object
        directors: NDArray
            An array of shape (n_dim, n_dim) that stores the directors of the Pose object
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
        Update the states of the Pose object

        Parameters
        ----------
        positions: NDArray
            The positions of the Pose objects. Expected shape is (n_dim,)
        directors: NDArray
            The directors of the Pose objects. Expected shape is (n_dim, n_dim)
        """
        self.spheres[0].update_states(positions)

        for i, cylinder in enumerate(self.cylinders):
            tip_position = positions + directors[:, i] * self.__unit_length
            cylinder.update_states(positions, tip_position)

            sphere = self.spheres[i + 1]
            sphere.update_states(tip_position)

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the Pose object

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the material update
        """
        for shperes in self.spheres:
            shperes.update_material(**kwargs)

        for cylinder in self.cylinders:
            cylinder.update_material(**kwargs)

    def update_keyframe(self, keyframe: int) -> None:
        """
        Set the keyframe for the pose object
        """
        for shperes in self.spheres:
            shperes.update_keyframe(keyframe)

        for cylinder in self.cylinders:
            cylinder.update_keyframe(keyframe)


if TYPE_CHECKING:
    data = {
        "positions": np.array([0.0, 0.0, 0.0]),
        "directors": np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
    }
    _: CompositeProtocol = Pose.create(data)
