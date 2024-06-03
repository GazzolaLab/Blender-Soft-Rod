from pathlib import Path

import bpy
from mathutils import Vector


def save(path: Path | str) -> bool:
    """
    TODO: documentation: one sentence summary of what the function does.

    Parameters
    ----------
    path : Path | str
        File path to save the file to.

    Returns
    -------
    bool
        Whether the file was successfully saved.
    """
    # Documentation
    # This function checks if the input is a Path or string and then saves the .blend file accordingly, raising an error if the input type is wrong.

    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise TypeError(
            f"Type of path should be either Path or str. Given: {type(path)}"
        )
    bpy.ops.wm.save_as_mainfile(filepath=str(path))
    print("Saved as .blend")
    return path.exists()


def reload(path: Path | str) -> bool:
    """
    TODO: documentation: one sentence summary of what the function does.

    Parameters
    ----------
    path : Path | str
        File path to reload the file from.

    Returns
    -------
    bool
        Whether the file was successfully reloaded.
    """

    if isinstance(path, Path):
        path = str(path)
    elif isinstance(path, str):
        pass
    else:
        raise TypeError(
            f"Type of path should be either Path or str. Given: {type(path)}"
        )

    # bpy.ops.wm.open_mainfile(filepath=path)
    # print("file opened")
    bpy.ops.wm.revert_mainfile()
    print("file reverted to last save")
    # What to return?

    print(bpy.context.active_object.location == Vector((0, 0, 0)))
    return bpy.context.active_object.location == Vector((0, 0, 0))
