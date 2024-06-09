__all__ = [
    "BlenderMeshInterfaceProtocol",
    "CompositeProtocol",
    "StackProtocol",
]

from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, Type, TypeVar
from typing_extensions import Self

from abc import ABC, abstractmethod

import bpy


class BlenderKeyframeManipulateProtocol(Protocol):
    def clear_animation(self) -> None: ...

    def set_keyframe(self, keyframe:int) -> None: ...

MeshDataType = dict[str, Any]
S = TypeVar("S", bound="BlenderMeshInterfaceProtocol")
P = ParamSpec("P")

class BlenderMeshInterfaceProtocol(BlenderKeyframeManipulateProtocol, Protocol):
    """
    This protocol defines the interface for Blender mesh objects.
    """

    # TODO: For future implementation
    # @property
    # def data(self): ...

    # @property
    # def material(self): ...

    @property
    def object(self) -> bpy.types.Object:
        """Returns associated Blender object."""

    @classmethod
    def create(cls: Type[S], states: MeshDataType) -> S:
        """Creates a new mesh object with the given states."""

    def update_states(self, *args: Any) -> bpy.types.Object:
        """Updates the mesh object with the given states."""

    # def update_material(self, material) -> None: ...  # TODO: For future implementation

class CompositeProtocol(BlenderMeshInterfaceProtocol, Protocol):
    @property
    def object(self) -> dict[str, list[bpy.types.Object]]:
        """Returns associated Blender object."""

D = TypeVar("D", bound="StackProtocol", covariant=True)
class StackProtocol(BlenderMeshInterfaceProtocol, Protocol[D]):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> D: ...

    @property
    def object(self) -> list[BlenderMeshInterfaceProtocol]:
        """Returns associated Blender object."""
