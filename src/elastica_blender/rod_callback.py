__all__ = ["BlenderRodCallback"]

from typing import Any

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

    def __init__(self, step_skip: int, *args: Any, **kwargs: Any) -> None:
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.keyframe = 0
        self.bpy_objs: bsr.Rod
        self.stop = False

    def make_callback(
        self, system: RodType, time: np.floating, current_step: int
    ) -> None:
        if self.stop or current_step % self.every != 0:
            return
        if (
            np.isnan(system.position_collection).any()
            or np.isnan(system.radius).any()
        ):
            self.stop = True
            return
        if current_step == 0:
            self.bpy_objs = bsr.Rod(system.position_collection, system.radius)
        else:
            self.bpy_objs.update_states(
                positions=system.position_collection,
                radii=system.radius,
            )
        self.bpy_objs.set_keyframe(self.keyframe)
        self.keyframe += 1
