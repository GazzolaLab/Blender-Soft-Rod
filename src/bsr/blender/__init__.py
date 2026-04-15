"""Compatibility Blender-focused exports for existing bsr workflows."""

from bsr._camera import Camera
from bsr._light import Light
from bsr.blender_commands.file import reload, save
from bsr.blender_commands.macros import (
    clear_materials,
    clear_mesh_objects,
    deselect_all,
    scene_update,
)
from bsr.frame import FrameManager
from bsr.geometry.composite.pose import Pose
from bsr.geometry.composite.rod import Rod, RodWithBox, RodWithCylinder
from bsr.geometry.composite.stack import RodStack, create_rod_collection
from bsr.geometry.primitives.pipe import BezierSplinePipe
from bsr.geometry.primitives.simple import Cylinder, Sphere
from bsr.viewport import find_area, set_view_distance

__all__ = [
    "BezierSplinePipe",
    "Camera",
    "Cylinder",
    "FrameManager",
    "Light",
    "Pose",
    "Rod",
    "RodStack",
    "RodWithBox",
    "RodWithCylinder",
    "Sphere",
    "clear_materials",
    "clear_mesh_objects",
    "create_rod_collection",
    "deselect_all",
    "find_area",
    "reload",
    "save",
    "scene_update",
    "set_view_distance",
]
