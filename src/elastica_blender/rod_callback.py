import bpy
import numpy as np
from elastica import CallBackBaseClass

from bsr.geometry import Cylinder, Sphere


class BlenderRodCallback(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.bpy_objs = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step == 0:
            self.build(system.position_collection)
        elif current_step % self.every == 0:
            self.update(
                positions=system.position_collection,
                time_step=current_step // self.every,
            )
        else:
            pass

    def build(self, positions):
        # New object
        for j in range(positions.shape[-1]):
            self.bpy_objs["sphere"].append(Sphere(positions[:, j]))
        for j in range(positions.shape[-1] - 1):
            self.bpy_objs["cylinder"].append(
                Cylinder(
                    self.bpy_objs["sphere"][j].obj.location,
                    self.bpy_objs["sphere"][j + 1].obj.location,
                )
            )

    def update(self, positions, time_step):
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
