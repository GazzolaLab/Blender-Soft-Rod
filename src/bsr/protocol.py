from typing import TYPE_CHECKING, Any, ParamSpec, Protocol, Type, TypeVar
from typing_extensions import Self

MeshDataType = dict[str, Any]
S = TypeVar("S", bound="BlenderMeshInterfaceProtocol")
P = ParamSpec("P")


class BlenderMeshInterfaceProtocol(Protocol):
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
