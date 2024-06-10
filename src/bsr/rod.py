__doc__ = """
Rod class for creating and updating rods in Blender
"""
__all__ = ["RodWithSphereAndCylinder", "Rod"]

from typing import TYPE_CHECKING

import bpy
import numpy as np

from .geometry import Cylinder, Sphere
from .mixin import KeyFrameControlMixin
from .protocol import CompositeProtocol


class RodWithSphereAndCylinder(KeyFrameControlMixin):
    """
    Rod class for managing visualization and rendering in Blender

    Parameters
    ----------
    positions : np.ndarray
        The positions of the sphere objects. Expected shape is (n_dim, n_nodes).
        n_dim = 3
    radii : np.ndarray
        The radii of the sphere objects. Expected shape is (n_nodes-1,).

    """

    def __init__(self, positions: np.ndarray, radii: np.ndarray) -> None:
        # check shape of positions and radii
        assert positions.ndim == 2, "positions must be 2D array"
        assert positions.shape[0] == 3, "positions must have 3 rows"
        assert radii.ndim == 1, "radii must be 1D array"
        assert (
            positions.shape[-1] == radii.shape[-1] + 1
        ), "radii must have n_nodes-1 elements"

        # create sphere and cylinder objects
        self.spheres: list[Sphere] = []
        self.cylinders: list[Cylinder] = []
        self._bpy_objs: dict[str, list[bpy.types.Object]] = {
            "sphere": self.spheres,
            "cylinder": self.cylinders,
        }

        self._build(positions, radii)

    @property
    def object(self) -> dict[str, list[bpy.types.Object]]:
        return self._bpy_objs

    @classmethod
    def create(
        cls, states: dict[str, np.ndarray]
    ) -> "RodWithSphereAndCylinder":
        rod = cls(states["positions"], states["radii"])
        return rod

    def _build(self, positions: np.ndarray, radii: np.ndarray) -> None:
        _radii = np.concatenate([radii, [0]])
        _radii[1:] += radii
        _radii[1:-1] /= 2.0
        for j in range(positions.shape[-1]):
            sphere = Sphere(positions[:, j], _radii[j])
            self.spheres.append(sphere)

        for j in range(radii.shape[-1]):
            cylinder = Cylinder(
                positions[:, j],
                positions[:, j + 1],
                radii[j],
            )
            self.cylinders.append(cylinder)

    def update_states(self, positions: np.ndarray, radii: np.ndarray) -> None:
        _radii = np.concatenate([radii, [0]])
        _radii[1:] += radii
        _radii[1:-1] /= 2.0
        for idx, sphere in enumerate(self.spheres):
            sphere.update_states(positions[:, idx], radii[idx])

        for idx, cylinder in enumerate(self.cylinders):
            cylinder.update_states(
                positions[:, idx], positions[:, idx + 1], radii[idx]
            )

    def set_keyframe(self, keyframe: int) -> None:
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
