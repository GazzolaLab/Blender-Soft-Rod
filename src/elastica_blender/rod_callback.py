import bpy
import numpy as np
from elastica import CallBackBaseClass
from elastica.typing import SystemType

import bsr
from bsr.geometry import Cylinder, Sphere


class BlenderRodCallback(CallBackBaseClass):
    """
    Call back function for continuum snake
    """

    def __init__(self, step_skip: int) -> None:
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.keyframe = 0
        self.bpy_objs = bsr.Rod()

    def make_callback(
        self, system: Systemtype, time: np.floating, current_step: int
    ):
        if current_step % self.every == 0:
            self.bpy_objs.update(
                keyframe=self.key_frame,
                positions=system.position_collection,
                radii=system.radius_collection,
            )
            self.key_frame += 1
