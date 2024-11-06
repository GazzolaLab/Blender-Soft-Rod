__all__ = ["BaseStack", "RodStack", "create_rod_collection"]

from typing import TYPE_CHECKING, Any, Protocol, Type, overload

import sys

# Check python version
if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

from collections.abc import Sequence

import bpy
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from bsr.geometry.composite.rod import Rod
from bsr.geometry.protocol import BlenderMeshInterfaceProtocol, StackProtocol
from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class BaseStack(Sequence, KeyFrameControlMixin):
    """
    A stack of objects that can be manipulated together.
    Internally, we use a list-like structure to store the objects.
    """

    DefaultType: Type

    def __init__(self) -> None:
        self._objs: list[BlenderMeshInterfaceProtocol] = []
        self._mats: list[BlenderMeshInterfaceProtocol] = []

    @overload
    def __getitem__(self, index: int, /) -> BlenderMeshInterfaceProtocol: ...

    @overload
    def __getitem__(
        self, index: slice, /
    ) -> list[BlenderMeshInterfaceProtocol]: ...

    def __getitem__(
        self, index: int | slice
    ) -> BlenderMeshInterfaceProtocol | list[BlenderMeshInterfaceProtocol]:
        return self._objs[index]

    def __len__(self) -> int:
        return len(self._objs)

    @property
    def material(self) -> list[BlenderMeshInterfaceProtocol]:
        """
        Returns the materials in the stack.
        """
        return self._mats

    @property
    def object(self) -> list[BlenderMeshInterfaceProtocol]:
        """
        Returns the objects in the stack.
        """
        return self._objs

    def update_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.
        """
        for obj in self._objs:
            obj.update_keyframe(keyframe)

    @classmethod
    def create(
        cls,
        states: dict[str, NDArray],
    ) -> Self:
        """
        Creates a stack of objects from the given states.
        """
        self = cls()
        keys = states.keys()
        lengths = [i.shape[0] for i in states.values()]
        assert len(set(lengths)) <= 1, "All states must have the same length"
        num_objects = lengths[0]

        for oidx in range(num_objects):
            state = {k: v[oidx] for k, v in states.items()}
            obj = self.DefaultType.create(state)
            self._objs.append(obj)
            self._mats.append(obj.material)
        return self

    def update_states(self, *variables: NDArray) -> None:
        """
        Updates the states of the objects.
        """
        if not all([v.shape[0] == len(self) for v in variables]):
            raise IndexError(
                "All variables must have the same length as the stack"
            )
        for idx in range(len(self)):
            self[idx].update_states(*[v[idx] for v in variables])

    def update_material(self, **kwargs: dict[str, NDArray]) -> None:
        """
        Updates the material of the objects.
        """
        for material_key, material_values in kwargs.items():
            assert isinstance(
                material_values, np.ndarray
            ), "Values of kwargs must be a numpy array"
            if material_values.shape[0] != len(self):
                raise IndexError(
                    "All values must have the same length as the stack"
                )
            for idx in range(len(self)):
                self[idx].update_material({material_key: material_values[idx]})


class RodStack(BaseStack):
    input_states = {"positions", "radii"}
    DefaultType: Type = Rod


# Alias for factory functions
create_rod_collection = RodStack.create


if TYPE_CHECKING:
    data: dict[str, NDArray] = {
        "positions": np.array([[[0, 0, 0], [1, 1, 1]]]),
        "radii": np.array([[1.0, 1.0]]),
    }
    _: StackProtocol = RodStack.create(data)
