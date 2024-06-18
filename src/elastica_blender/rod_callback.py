__all__ = ["BlenderRodCallback"]

import bpy
import numpy as np
from elastica import CallBackBaseClass
from elastica.typing import RodType

import bsr
from bsr.geometry import Cylinder, Sphere


class BlenderRodCallback(CallBackBaseClass):
    """
    PyElastica callback to save rod state to Blender.
    """

    def __init__(self, step_skip: int) -> None:
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.keyframe = 0
        self.bpy_objs: bsr.Rod

    def make_callback(
        self, system: RodType, time: np.floating, current_step: int
    ) -> None:
        if current_step % self.every == 0:
            if current_step == 0:
                self.bpy_objs = bsr.Rod(
                    system.position_collection, system.radius
                )
            else:
                self.bpy_objs.update_states(
                    positions=system.position_collection,
                    radii=system.radius,
                )
            self.bpy_objs.set_keyframe(self.keyframe)
            self.keyframe += 1
