from collections.abc import Iterator

import bpy


def iter_action_fcurves(action: bpy.types.Action) -> Iterator[bpy.types.FCurve]:
    """
    Iterate over f-curves for both Blender 4.x and Blender 5.x action data models.
    """
    # Blender 4.x actions expose fcurves directly.
    fcurves = getattr(action, "fcurves", None)
    if fcurves is not None:
        yield from fcurves
        return

    # Blender 5.x uses layered actions with keyframe strips and channel bags.
    for layer in getattr(action, "layers", ()):
        for strip in getattr(layer, "strips", ()):
            for channelbag in getattr(strip, "channelbags", ()):
                for fcurve in getattr(channelbag, "fcurves", ()):
                    yield fcurve
