from typing import Optional

import bpy


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
