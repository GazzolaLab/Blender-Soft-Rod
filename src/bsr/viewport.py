from typing import Optional

import bpy


def find_area(area_type: str) -> bpy.types.Area:
    """
    Return the area of the given type in the current screen

    Parameters
    ----------
    area_type : str
        The type of the area to find and return, for example: "VIEW_3D".
    """
    try:
        for area in bpy.data.window_managers[0].windows[0].screen.areas:
            if area.type == area_type:
                return area
        raise ValueError(f"No area of type {area_type} found")
    except:
        assert False, "No area found"


def set_view_distance(distance: float) -> None:
    """
    Set the view distance of the 3D view

    Parameters
    ----------
    distance : float
        The view distance to set.
    """
    assert (
        isinstance(distance, (int, float)) and distance > 0
    ), "distance must be a positive number"
    area_view_3d = find_area("VIEW_3D")
    try:
        region_3d = area_view_3d.spaces[0].region_3d
        region_3d.view_distance = distance
    except:
        assert False, "No 3D view or 3D region found"
