from pathlib import Path

import bpy
from mathutils import Vector


def save(path: Path | str) -> bool:
    # Documentation
    # Checks if the input is a Path or string and then saves the .blend file accordingly, raising an error if the input type is wrong.
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise TypeError(
            f"Type of path should be either Path or str. Given: {type(path)}"
        )
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    return path.exists()


def reload(path: Path | str) -> bool:
    # Documentation
    # Reloads (resets) the file to previous save condition, and raises error if filename is not found.
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise TypeError(
            f"Type of path should be either Path or str. Given: {type(path)}"
        )
    if Path(path).exists() == False:
        raise FileNotFoundError("This file does not exist")
    bpy.ops.wm.revert_mainfile()
    return bpy.context.active_object.location == Vector((0, 0, 0))
