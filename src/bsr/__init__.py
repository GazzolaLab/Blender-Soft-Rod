from typing import Optional

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


class Frame:
    def __init__(self, frame: int = 0):
        self.__frame = frame

    def update(self, forwardframe: int = 1) -> None:
        assert (
            isinstance(forwardframe, int) and forwardframe > 0
        ), "forwardframe must be a positive integer"
        self.__frame += forwardframe

    @property
    def current_frame(self) -> int:
        return self.__frame

    @current_frame.setter
    def current_frame(self, frame: int) -> None:
        assert (
            isinstance(frame, int) and frame >= 0
        ), "frame must be a positive integer or 0"
        self.__frame = frame

    def set_frame_end(self, frame: Optional[int] = None) -> None:
        if frame is None:
            frame = self.__frame
        else:
            assert (
                isinstance(frame, int) and frame >= 0
            ), "frame must be a positive integer or 0"

        import bpy

        bpy.context.scene.frame_end = frame


frame = Frame()
