__doc__ = """
Rod class for creating and updating rods in Blender
"""
__all__ = ["Rod"]

import numpy as np

from .geometry import Cylinder, Sphere


# TODO
class Rod:
    """
    Rod class for managing visualization and rendering in Blender
    """

    def __init__(self):
        self.bpy_objs = None

    def clear(self):
        raise NotImplementedError("Not yet implemented")

    def build(self, positions: np.ndarray, radii: np.ndarray):
        # TODO: Refactor
        for j in range(positions.shape[-1]):
            sphere = Sphere(positions[:, j])
            self.bpy_objs["sphere"].append(sphere)
            sphere.obj.keyframe_insert(data_path="location", frame=0)

        for j in range(positions.shape[-1] - 1):
            cylinder = Cylinder(
                self.bpy_objs["sphere"][j].obj.location,
                self.bpy_objs["sphere"][j + 1].obj.location,
            )
            self.bpy_objs["cylinder"].append(cylinder)
            cylinder.obj.keyframe_insert(data_path="location", frame=0)

    def update(self, keyframe: int, positions: np.ndarray, radii: np.ndarray):
        if self.bpy_objs is None:
            self.bpy_objs = {"sphere": [], "cylinder": []}
            self.build(positions, radii)
            return
        # TODO: Refactor
        # update all sphere and cylinder positions and write object to keyframe
        for idx, sphere in enumerate(self.bpy_objs["sphere"]):
            sphere.update_position(positions[:, idx])
            sphere.obj.keyframe_insert(data_path="location", frame=time_step)

        for idx, cylinder in enumerate(self.bpy_objs["cylinder"]):
            cylinder.update_position(
                self.bpy_objs["sphere"][idx].obj.location,
                self.bpy_objs["sphere"][idx + 1].obj.location,
            )
            cylinder.obj.keyframe_insert(data_path="location", frame=time_step)
            cylinder.obj.keyframe_insert(
                data_path="rotation_euler", frame=time_step
            )
            cylinder.obj.keyframe_insert(data_path="scale", frame=time_step)
            cylinder.mat.keyframe_insert(
                data_path="diffuse_color", frame=time_step
            )
