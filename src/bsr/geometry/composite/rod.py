__doc__ = """
Rod class for creating and updating rods in Blender
"""
__all__ = ["RodWithSphereAndCylinder", "Rod"]

from typing import TYPE_CHECKING, Any

from collections import defaultdict

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.geometry.primitives.simple import Cylinder, Sphere
from bsr.geometry.protocol import CompositeProtocol
from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class RodWithSphereAndCylinder(KeyFrameControlMixin):
    """
    Rod class for managing visualization and rendering in Blender

    Parameters
    ----------
    positions : NDArray
        The positions of the sphere objects. Expected shape is (n_dim, n_nodes).
        n_dim = 3
    radii : NDArray
        The radii of the sphere objects. Expected shape is (n_nodes-1,).

    """

    input_states = {"positions", "radii"}

    def __init__(self, positions: NDArray, radii: NDArray) -> None:
        # create sphere and cylinder objects
        self.spheres: list[Sphere] = []
        self.cylinders: list[Cylinder] = []
        self._bpy_objs: dict[str, list[bpy.types.Object]] = {
            "sphere": self.spheres,
            "cylinder": self.cylinders,
        }

        # create sphere and cylinder materials
        self.spheres_material: list[bpy.types.Material] = []
        self.cylinders_material: list[bpy.types.Material] = []
        self._bpy_materials: dict[str, list[bpy.types.Material]] = {
            "sphere": self.spheres_material,
            "cylinder": self.cylinders_material,
        }

        self._build(positions, radii)

    @property
    def material(self) -> dict[str, list[bpy.types.Material]]:
        """
        Return the dictionary of Blender materials: sphere and cylinder
        """
        return self._bpy_materials

    @property
    def object(self) -> dict[str, list[bpy.types.Object]]:
        """
        Return the dictionary of Blender objects: sphere and cylinder
        """
        return self._bpy_objs

    @classmethod
    def create(cls, states: dict[str, NDArray]) -> "RodWithSphereAndCylinder":
        """
        Create a Rod object from the given states

        States must have the following keys: positions(n_nodes, 3), radii(n_nodes-1,)
        """
        rod = cls(**states)
        return rod

    def _build(self, positions: NDArray, radii: NDArray) -> None:
        _radii = np.concatenate([radii, [0]])
        _radii[1:] += radii
        _radii[1:-1] /= 2.0
        for j in range(positions.shape[-1]):
            sphere = Sphere(positions[:, j], _radii[j])
            self.spheres.append(sphere)
            self.spheres_material.append(sphere.material)

        for j in range(radii.shape[-1]):
            cylinder = Cylinder(
                positions[:, j],
                positions[:, j + 1],
                radii[j],
            )
            self.cylinders.append(cylinder)
            self.cylinders_material.append(cylinder.material)

    def update_states(self, positions: NDArray, radii: NDArray) -> None:
        """
        Update the states of the rod object

        Parameters
        ----------
        positions : NDArray
            The positions of the sphere objects. Expected shape is (n_nodes, 3).
        radii : NDArray
            The radii of the sphere objects. Expected shape is (n_nodes-1,).
        """
        # check shape of positions and radii
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert radii.ndim == 1, "radii must be 1D array"
        assert (
            positions.shape[-1] == radii.shape[-1] + 1
        ), "radii must have n_nodes-1 elements"

        _radii = np.concatenate([radii, [0]])
        _radii[1:] += radii
        _radii[1:-1] /= 2.0
        for idx, sphere in enumerate(self.spheres):
            sphere.update_states(positions[:, idx], _radii[idx])

        for idx, cylinder in enumerate(self.cylinders):
            cylinder.update_states(
                positions[:, idx], positions[:, idx + 1], _radii[idx]
            )

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the rod object
        """
        for shperes in self.spheres:
            shperes.update_material(**kwargs)

        for cylinder in self.cylinders:
            cylinder.update_material(**kwargs)

    def set_keyframe(self, keyframe: int) -> None:
        """
        Set keyframe for the rod object
        """
        for idx, sphere in enumerate(self.spheres):
            sphere.set_keyframe(keyframe)

        for idx, cylinder in enumerate(self.cylinders):
            cylinder.set_keyframe(keyframe)


# Alias
Rod = RodWithSphereAndCylinder

if TYPE_CHECKING:
    data = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]),
        "radii": np.array([1.0, 1.0]),
    }
    _: CompositeProtocol = RodWithSphereAndCylinder.create(data)
