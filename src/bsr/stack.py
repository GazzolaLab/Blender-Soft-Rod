__all__ = ["RodStack", "create_rod_collection"]

from typing import TYPE_CHECKING, Protocol, Any
from typing_extensions import Self

import numpy as np

import bpy

from .protocol import StackProtocol
from .rod import Rod
from .mixin import KeyFrameControlMixin


# TODO
class RodStack(KeyFrameControlMixin):
    def __init__(self) -> None:
        self._rods: list["Rod"] = []

    def __getitem__(self, index: int) -> "Rod":
        return self._rods[index]

    def __len__(self) -> int:
        return len(self._rods)

    @property
    def object(self) -> list[bpy.types.Object]:
        return self._rods

    @classmethod
    def create(cls, states: dict[str, np.ndarray]) -> Self:
        return cls()

    def update_states(self, *args: Any) -> None:
        pass

    @classmethod
    def create_collection(
        cls, num_rods: int, num_nodes: int
    ) -> Self:
        return cls()

    def update_history(
        self, keyframes: np.ndarray, position: np.ndarray, radius: np.ndarray
    ) -> None:
        pass

    def set_keyframe(self, keyframe:int) -> None:
        # TODO
        pass


# Alias for factory functions
create_rod_collection = RodStack.create_collection


if TYPE_CHECKING:
    _: StackProtocol = RodStack()
