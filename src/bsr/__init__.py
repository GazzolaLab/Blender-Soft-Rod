from typing import Final, Optional

import sys
from importlib import metadata as importlib_metadata

import bpy

from ._camera import Camera
from ._light import Light

# Exposed functions and classes (API)
# Note: These should not be imported within the package to avoid circular imports
from .blender_commands.file import reload, save
from .blender_commands.macros import (
    clear_materials,
    clear_mesh_objects,
    deselect_all,
    scene_update,
)
from .frame import FrameManager
from .geometry.composite.pose import Pose
from .geometry.composite.rod import Rod
from .geometry.composite.stack import RodStack, create_rod_collection
from .geometry.primitives.simple import Cylinder, Sphere
from .viewport import find_area, set_view_distance


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: Final[str] = get_version()
frame_manager = FrameManager()
camera = Camera()
light = Light()
