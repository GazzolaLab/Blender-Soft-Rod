from typing import Final, cast

from importlib import import_module
from importlib import metadata as importlib_metadata


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: Final[str] = get_version()

_LAZY_EXPORTS: dict[str, tuple[str, str, bool]] = {
    "Camera": ("bsr._camera", "Camera", False),
    "Light": ("bsr._light", "Light", False),
    "reload": ("bsr.blender_commands.file", "reload", False),
    "save": ("bsr.blender_commands.file", "save", False),
    "clear_materials": (
        "bsr.blender_commands.macros",
        "clear_materials",
        False,
    ),
    "clear_mesh_objects": (
        "bsr.blender_commands.macros",
        "clear_mesh_objects",
        False,
    ),
    "deselect_all": ("bsr.blender_commands.macros", "deselect_all", False),
    "scene_update": ("bsr.blender_commands.macros", "scene_update", False),
    "FrameManager": ("bsr.frame", "FrameManager", False),
    "Pose": ("bsr.geometry.composite.pose", "Pose", False),
    "Rod": ("bsr.geometry.composite.rod", "Rod", False),
    "RodWithBox": ("bsr.geometry.composite.rod", "RodWithBox", False),
    "RodWithCylinder": ("bsr.geometry.composite.rod", "RodWithCylinder", False),
    "RodStack": ("bsr.geometry.composite.stack", "RodStack", False),
    "create_rod_collection": (
        "bsr.geometry.composite.stack",
        "create_rod_collection",
        False,
    ),
    "BezierSplinePipe": (
        "bsr.geometry.primitives.pipe",
        "BezierSplinePipe",
        False,
    ),
    "Cylinder": ("bsr.geometry.primitives.simple", "Cylinder", False),
    "Sphere": ("bsr.geometry.primitives.simple", "Sphere", False),
    "find_area": ("bsr.viewport", "find_area", False),
    "set_view_distance": ("bsr.viewport", "set_view_distance", False),
    "frame_manager": ("bsr.frame", "FrameManager", True),
    "camera": ("bsr._camera", "Camera", True),
    "light": ("bsr._light", "Light", True),
}

__all__ = ["version", *_LAZY_EXPORTS.keys()]


def __getattr__(name: str) -> object:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name, instantiate = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    if instantiate:
        value = value()
    globals()[name] = value
    return cast(object, value)
