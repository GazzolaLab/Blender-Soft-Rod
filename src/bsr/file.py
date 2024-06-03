from pathlib import Path

import bpy


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
    if isinstance(path, Path):
        pass
    elif isinstance(path, str):
        path = Path(path)
    else:
        raise TypeError("Type of path should be either Path or str.")

    if path.suffix != ".blend":
        path = path.with_suffix(".blend")
    
    bpy.ops.wm.save_as_mainfile(filepath=path.as_posix())
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
    raise NotImplementedError("Not implemented yet.")
    return False
