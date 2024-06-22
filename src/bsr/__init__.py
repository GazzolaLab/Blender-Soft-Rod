import sys
from importlib import metadata as importlib_metadata

# Exposed functions and classes (API)
# Note: These should not be imported within the package to avoid circular imports
from .blender_commands.file import reload, save
from .blender_commands.macros import (
    clear_materials,
    clear_mesh_objects,
    scene_update,
)
from .geometry.composite.rod import Rod
from .geometry.composite.stack import RodStack, create_rod_collection
from .geometry.primitives.simple import Cylinder, Sphere


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
