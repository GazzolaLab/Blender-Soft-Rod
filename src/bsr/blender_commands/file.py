""" file.py contains functions useful in handling .blend files using bpy """

from pathlib import Path

import bpy
from mathutils import Vector


def save(path: Path | str) -> bool:
    """
    Saves working blender environment to a blender file.

    Parameters
    ----------
    path: Path | str
        File path to save the file to.

    Returns
    -------
    bool:
        Whether the file was successfully saved.
        In this case checked if the filepath exists.

    Raises
    ------
    TypeError
        If the type of the path parameter is not a Path or string

    Examples
    --------
    >>> import bsr

    ... # work on blender file

    >>> blender_path: str | Path
    >>> bsr.save(blender_path)
    """
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


def reload(path: Path | str) -> None:
    """
    Reloads the blender file to most recently saved state.

    Parameters
    ----------
    path: Path | str
        File path to save the file to.

    Raises
    ------
    TypeError
        If the type of the path parameter is not a Path or string

    FileNotFoundError
        If the passed-in filepath does not exist

    Examples
    --------
    >>> import bsr

    ... # work on blender file

    >>> blender_path: str | Path
    >>> bsr.reload(blender_path)
    """
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
