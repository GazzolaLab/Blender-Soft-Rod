from typing import Final, Optional

import sys
from importlib import metadata as importlib_metadata

import bpy

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


version: Final[str] = get_version()


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
        bpy.context.scene.frame_end = frame


frame = Frame()


def find_area(area_type: str) -> Optional[bpy.types.Area]:
    try:
        for area in bpy.data.window_managers[0].windows[0].screen.areas:
            if area.type == area_type:
                return area
        return None
    except:
        return None


def set_view_distance(distance: float) -> None:
    assert (
        isinstance(distance, (int, float)) and distance > 0
    ), "distance must be a positive number"
    area_view_3d = find_area("VIEW_3D")
    region_3d = area_view_3d.spaces[0].region_3d
    region_3d.view_distance = distance
