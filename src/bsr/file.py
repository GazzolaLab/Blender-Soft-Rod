""" file.py contains functions useful in handling .blend files using bpy """

from pathlib import Path

import bpy
from mathutils import Vector


def save(path: Path | str) -> bool:
    """
    Saves .blend file.

    To use this function, pass in a filepath in either Path or string format.
    If the input is a string, the function will convert it back into a path in a conditional.
    If the input is not a Path or string, a TypeError will be raised.
    Once all valid input types are in path format, they are converted back into a string.
    These are then used as the filepath parameter for the save_as_mainfile operator.
    This operator saves a .blend file to the filepath that is passed in.

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
    >>> path = /Users/rohitharish/Downloads/test_path
    >>> bpy.ops.wm.save_as_mainfile(filepath=str(path))
    #Saves file as test_path.blend
    >>> return path.exists()
    True

    >>> path = "/Users/rohitharish/Downloads/test_path"
    >>> path = Path(path)
    /Users/rohitharish/Downloads/test_path
    >>> bpy.ops.wm.save_as_mainfile(filepath=str(path))
    #Saves file as test_path.blend
    >>> return path.exists()
    True

    >>> path = 50
    >>> raise TypeError(
            f"Type of path should be either Path or str. Given: {type(path)}"
        )
    Type of path should be either Path or str. Given: {int(path)}
    >>> return path.exists()
    False
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
    Reloads the .blend file to most recent saved state.

    To use this function, pass in a filepath in either Path or string format.
    If the input is a string, the function will convert it back into a path in a conditional.
    If the input is not a Path or string, a TypeError will be raised.
    If the filepath does not exist, a FileNotFoundError will be raised.
    If these conditions are all passed, the file is reloaded with the revert_mainfile operator.

    Parameters
    ----------
    path: Path | str
        File path to save the file to.

    Returns
    -------
    bool:
        Whether the file was successfully reloaded.
        In this case doesn't need to return anything.

    Raises
    ------
    TypeError
        If the type of the path parameter is not a Path or string

    FileNotFoundError
        If the passed-in filepath does not exist

    Examples
    --------
    >>> path = /Users/rohitharish/Downloads/test_path
    >>> bpy.ops.wm.revert_mainfile()
    #Reloads file to last saved state

    >>> path = "/Users/rohitharish/Downloads/test_path"
    >>> path = Path(path)
    /Users/rohitharish/Downloads/test_path
    >>> bpy.ops.wm.revert_meainfile()
    #Reloads file to last saved state

    >>> path = 50
    >>> raise TypeError(
            f"Type of path should be either Path or str. Given: {type(path)}"
        )
    Type of path should be either Path or str. Given: {int(path)}

    >>> path = /Users/rohitharish/Downloads/non_existent_file.blend
    >>> raise FileNotFoundError("This file does not exist")
    This file does not exist

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
