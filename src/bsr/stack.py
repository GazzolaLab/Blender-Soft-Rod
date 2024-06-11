__all__ = ["RodStack", "create_rod_collection"]

from typing import TYPE_CHECKING, Any, Protocol, Type
from typing_extensions import Self

from collections.abc import ABC, Sequence
from functools import partial

import bpy
import numpy as np
from tqdm import tqdm

from .mixin import KeyFrameControlMixin
from .protocol import BlenderMeshInterfaceProtocol, StackProtocol
from .rod import Rod
from .typing import RodType


class BaseStack(Sequence, KeyFrameControlMixin, ABC):
    def __init__(self) -> None:
        self._objs: list[BlenderMeshInterfaceProtocol] = []

    def __getitem__(self, index: int) -> BlenderMeshInterfaceProtocol:
        return self._objs[index]

    def __len__(self) -> int:
        return len(self._objs)

    @property
    def object(self) -> list[bpy.types.Object]:
        return self._objs

    def set_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.
        """
        for rod in self._objs:
            rod.set_keyframe(keyframe)

    @classmethod
    def create(
        cls,
        states: dict[str, np.ndarray],
        object_type: Type[BlenderMeshInterfaceProtocol],
    ) -> Self:
        self = cls()
        keys = states.keys()
        lengths = [i.shape[0] for i in states.values()]
        assert len(set(lengths)) <= 1, "All states must have the same length"
        num_objects = lengths[0]

        for oidx in range(num_objects):
            state = {k: v[oidx] for k, v in states.items()}
            obj = object_type.create(state)
            self._objs.append(obj)
        return self


class RodStack(BaseStack):

    def update_states(self, position: np.ndarray, radius: np.ndarray) -> None:
        """
        Updates the states of the rod objects.
        """
        for idx in range(len(self)):
            self[idx].update_states(position[idx], radius[idx])


# Alias for factory functions
create_rod_collection = partial(RodStack.create, object_type=Rod)


if TYPE_CHECKING:
    data = {
        "positions": np.array([[[0, 0, 0], [1, 1, 1]]]),
        "radii": np.array([[1.0, 1.0]]),
    }
    _: StackProtocol = RodStack.create(data, object_type=Rod)
