import sys
from importlib import metadata as importlib_metadata

# Exposed functions and classes (API)
# Note: These should not be imported within the package to avoid circular imports
from .blender_commands.file import save, reload
from .blender_commands.macros import clear_mesh_objects, scene_update, clear_materials
from .geometry.primitives.simple import (
    Sphere,
    Cylinder,
)
from .geometry.composite.rod import Rod
from .geometry.composite.stack import RodStack, create_rod_collection


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
