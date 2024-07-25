"""
Created on Jul 25, 2024
@author: Heng-Sheng (Hanson) Chang
"""

from dataclasses import dataclass

import elastica as ea
import numpy as np
from numba import njit

import bsr


@njit(cache=True)  # type: ignore
def material_to_lab(
    directors: np.ndarray, material_vectors: np.ndarray
) -> np.ndarray:
    blocksize = material_vectors.shape[1]
    lab_vectors = np.zeros((3, blocksize))
    for n in range(blocksize):
        for i in range(3):
            for j in range(3):
                lab_vectors[i, n] += directors[j, i, n] * material_vectors[j, n]
    return lab_vectors


class BasicCallBackBaseClass(ea.CallBackBaseClass):
    def __init__(self, step_skip: int):
        super().__init__()
        self.every = step_skip
        self.stop = False

    def make_callback(
        self, system: ea.CosseratRod, time: float, current_step: int
    ) -> None:
        if self.stop or current_step % self.every != 0:
            return
        if (
            np.isnan(system.position_collection).any()
            or np.isnan(system.radius).any()
        ):
            self.stop = True
            return
        self.save_params(system, time)

    def save_params(
        self,
    ):
        return NotImplementedError


@dataclass
class BR2Property:
    radii: np.ndarray
    bending_actuation_position: np.ndarray
    rotation_CW_actuation_position: np.ndarray
    rotation_CCW_actuation_position: np.ndarray


class BR2BpyObj:
    def __init__(
        self,
        property: BR2Property,
        default_centerline_position: np.ndarray,
        default_centerline_director: np.ndarray,
    ):
        self.property = property
        self.radii = property.radii
        self.bending_actuation = bsr.Rod(
            positions=self.compute_actuation_position(
                actuation_position=property.bending_actuation_position,
                centerline_position=default_centerline_position,
                centerline_director=default_centerline_director,
            ),
            radii=self.radii,
        )
        self.rotation_CW_actuation = bsr.Rod(
            positions=self.compute_actuation_position(
                actuation_position=property.rotation_CW_actuation_position,
                centerline_position=default_centerline_position,
                centerline_director=default_centerline_director,
            ),
            radii=self.radii,
        )
        self.rotation_CCW_actuation = bsr.Rod(
            positions=self.compute_actuation_position(
                actuation_position=property.rotation_CCW_actuation_position,
                centerline_position=default_centerline_position,
                centerline_director=default_centerline_director,
            ),
            radii=self.radii,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute_actuation_position(
        actuation_position: np.ndarray,
        centerline_position: np.ndarray,
        centerline_director: np.ndarray,
    ) -> np.ndarray:
        centerline_mid_position: np.ndarray = 0.5 * (
            centerline_position[:, :-1] + centerline_position[:, 1:]
        )
        actuation_position: np.ndarray = (
            centerline_mid_position
            + material_to_lab(
                centerline_director,
                actuation_position,
            )
        )
        return actuation_position

    def update_states(
        self,
        centerline_position: np.ndarray,
        centerline_director: np.ndarray,
    ):
        self.bending_actuation.update_states(
            positions=self.compute_actuation_position(
                actuation_position=self.property.bending_actuation_position,
                centerline_position=centerline_position,
                centerline_director=centerline_director,
            ),
            radii=self.radii,
        )
        self.rotation_CW_actuation.update_states(
            positions=self.compute_actuation_position(
                actuation_position=self.property.rotation_CW_actuation_position,
                centerline_position=centerline_position,
                centerline_director=centerline_director,
            ),
            radii=self.radii,
        )
        self.rotation_CCW_actuation.update_states(
            positions=self.compute_actuation_position(
                actuation_position=self.property.rotation_CCW_actuation_position,
                centerline_position=centerline_position,
                centerline_director=centerline_director,
            ),
            radii=self.radii,
        )

    def set_keyframe(self, keyframe: int) -> None:
        self.bending_actuation.set_keyframe(keyframe)
        self.rotation_CW_actuation.set_keyframe(keyframe)
        self.rotation_CCW_actuation.set_keyframe(keyframe)


class BlenderBR2CallBack(BasicCallBackBaseClass):
    def __init__(
        self,
        step_skip: int,
        property: BR2Property,
        system: ea.CosseratRod,
    ):
        super().__init__(step_skip=step_skip)
        self.bpy_objs: BR2BpyObj = BR2BpyObj(
            property=property,
            default_centerline_position=system.position_collection,
            default_centerline_director=system.director_collection,
        )
        self.keyframe = 0

    def save_params(self, system: ea.CosseratRod, time: float) -> None:
        self.bpy_objs.update_states(
            centerline_position=system.position_collection,
            centerline_director=system.director_collection,
        )
        self.bpy_objs.set_keyframe(self.keyframe)
        self.keyframe += 1
