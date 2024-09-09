__all__ = [
    "BlenderMeshInterfaceProtocol",
    "CompositeProtocol",
    "StackProtocol",
]

from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    Protocol,
    Type,
    TypeAlias,
    TypeVar,
)
from typing_extensions import Self

from abc import ABC, abstractmethod

import bpy
from numpy.typing import NDArray

from bsr.tools.protocol import BlenderKeyframeManipulateProtocol

MeshDataType: TypeAlias = dict[str, Any]
S = TypeVar("S", bound="BlenderMeshInterfaceProtocol")


class BlenderMeshInterfaceProtocol(BlenderKeyframeManipulateProtocol, Protocol):
    """
    This protocol defines the interface for Blender mesh objects.
    """

    input_states: set[str]

    # TODO: For future implementation
    # @property
    # def data(self): ...

    @property
    def material(self) -> Any:
        """Returns the material of the Blender mesh."""

    @property
    def object(self) -> Any:
        """Returns associated Blender object."""

    @classmethod
    def create(cls: Type[S], states: MeshDataType) -> S:
        """Creates a new mesh object with the given states."""

    def update_states(self, *args: Any) -> None:
        """Updates the mesh object with the given states."""

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """Updates the material of the mesh object."""


class CompositeProtocol(BlenderMeshInterfaceProtocol, Protocol):
    @property
    def object(
        self,
    ) -> dict[str, list[BlenderMeshInterfaceProtocol | bpy.types.Object]]:
        """Returns associated Blender object."""


class StackProtocol(BlenderMeshInterfaceProtocol, Protocol):
    DefaultType: Type

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> BlenderMeshInterfaceProtocol: ...

    @property
    def object(self) -> list[BlenderMeshInterfaceProtocol]:
        """Returns associated Blender object."""
