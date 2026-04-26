from __future__ import annotations

from dataclasses import dataclass, field

import elastica as ea
import numpy as np
from numba import njit


class SegmentExtensionActuation(ea.NoForces):
    def __init__(
        self,
        start_index: int,
        end_index: int,
        original_shear_matrix: np.ndarray,
        original_bend_matrix: np.ndarray,
        amplitude: callable,
    ) -> None:
        super().__init__()
        self.start_index = int(start_index)
        self.end_index = int(end_index)
        self.original_shear_matrix = original_shear_matrix[..., :end_index]
        self.original_bend_matrix = original_bend_matrix[..., :end_index]
        self.amplitude = amplitude

    def apply_forces(self, system: object, time: float = 0.0) -> None:
        stretch_magnitude, stiffness_magnitude, bend_magnitude = self.amplitude()
        self._apply_segment_extension_force(
            self.start_index,
            self.end_index,
            self.original_shear_matrix,
            self.original_bend_matrix,
            system.rest_sigma,
            system.rest_kappa,
            system.shear_matrix,
            system.bend_matrix,
            stretch_magnitude,
            bend_magnitude,
            (1.0 + stiffness_magnitude),
        )

    @staticmethod
    @njit(cache=True, fastmath=True)
    def _apply_segment_extension_force(
        start_index: int,
        end_index: int,
        original_shear_matrix: np.ndarray,
        original_bend_matrix: np.ndarray,
        rest_sigma: np.ndarray,
        rest_kappa: np.ndarray,
        shear_matrix: np.ndarray,
        bend_matrix: np.ndarray,
        stretch_scale: float,
        bend_scale: float,
        stiffness_scale: float,
    ) -> None:
        rest_sigma[2, :end_index] = stretch_scale
        rest_kappa[0, :end_index] = bend_scale
        shear_matrix[0, 0, :end_index] = (
            original_shear_matrix[0, 0, :] * stiffness_scale
        )
        shear_matrix[1, 1, :end_index] = (
            original_shear_matrix[1, 1, :] * stiffness_scale
        )
        bend_matrix[0, 0, :end_index] = original_bend_matrix[0, 0, :] * stiffness_scale
        bend_matrix[1, 1, :end_index] = original_bend_matrix[1, 1, :] * stiffness_scale


@dataclass(slots=True)
class CrawlingPolicy:
    _T_L: float = 2.4
    stiffness_center: float = 0.33
    stiffness_deviation: float = 0.2
    stiffness_scale: float = 0.0
    extension_center: float = 0.33
    extension_deviation: float = 0.10
    extension_scale: float = 0.5
    contraction_center: float = 0.66
    contraction_deviation: float = 0.20
    contraction_scale: float = 0.2
    base_suction_center: float = 0.33
    base_suction_deviation: float = 0.2
    base_suction_scale: float = 0.7
    middle_suction_center: float = 0.66
    middle_suction_deviation: float = 0.20
    middle_suction_scale: float = 0.7
    bend_center: float = 0.5
    bend_deviation: float = 0.10
    bend_scale: float = 0.0

    PARAMETER_NAMES = (
        "stiffness_center",
        "stiffness_deviation",
        "stiffness_scale",
        "extension_center",
        "extension_deviation",
        "extension_scale",
        "contraction_center",
        "contraction_deviation",
        "contraction_scale",
        "base_suction_center",
        "base_suction_deviation",
        "base_suction_scale",
        "middle_suction_center",
        "middle_suction_deviation",
        "middle_suction_scale",
        "bend_center",
        "bend_deviation",
        "bend_scale",
    )

    @classmethod
    def vector_size(cls) -> int:
        return len(cls.PARAMETER_NAMES)

    @classmethod
    def lower_bounds(cls) -> np.ndarray:
        """Per-arm box constraints for CMA-ES / clipping (phase in [0, 1], positive deviations)."""
        return np.asarray(
            [
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                0.0,
                0.0,
                1.0e-6,
                -0.5,
            ],
            dtype=np.float64,
        )

    @classmethod
    def upper_bounds(cls) -> np.ndarray:
        return np.asarray(
            [
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                1.0,
                1.0,
                0.5,
                0.5,
            ],
            dtype=np.float64,
        )

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in self.PARAMETER_NAMES],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(cls, values: np.ndarray, T_L: float = 2.4) -> "CrawlingPolicy":
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        kwargs = {name: float(value) for name, value in zip(cls.PARAMETER_NAMES, flat)}
        kwargs["_T_L"] = float(T_L)
        return cls(**kwargs)


@dataclass(slots=True)
class OctoArmPolicy:
    T_L: float = 2.4
    arm_policies: tuple[CrawlingPolicy, ...] = field(
        default_factory=lambda: tuple(CrawlingPolicy() for _ in range(8))
    )

    @classmethod
    def num_arms(cls) -> int:
        return 8

    @classmethod
    def vector_size(cls) -> int:
        return cls.num_arms() * CrawlingPolicy.vector_size()

    @classmethod
    def lower_bounds(cls) -> np.ndarray:
        return np.tile(CrawlingPolicy.lower_bounds(), cls.num_arms())

    @classmethod
    def upper_bounds(cls) -> np.ndarray:
        return np.tile(CrawlingPolicy.upper_bounds(), cls.num_arms())

    @classmethod
    def default(cls, T_L: float = 2.4) -> "OctoArmPolicy":
        return cls(
            T_L=float(T_L),
            arm_policies=tuple(CrawlingPolicy() for _ in range(cls.num_arms())),
        )

    def to_vector(self) -> np.ndarray:
        return np.concatenate(
            [policy.to_vector() for policy in self.arm_policies]
        ).astype(np.float64)

    @classmethod
    def from_vector(cls, values: np.ndarray, T_L: float = 2.4) -> "OctoArmPolicy":
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        arm_width = CrawlingPolicy.vector_size()
        arms = []
        for arm_index in range(cls.num_arms()):
            start = arm_index * arm_width
            arms.append(
                CrawlingPolicy.from_vector(flat[start : start + arm_width], T_L=T_L)
            )
        return cls(T_L=float(T_L), arm_policies=tuple(arms))


def idle_policy_like(policy: OctoArmPolicy) -> OctoArmPolicy:
    idle_arms = []
    for arm_policy in policy.arm_policies:
        values = arm_policy.to_vector().copy()
        values[[2, 5, 8, 11, 14, 17]] = 0.0
        idle_arms.append(CrawlingPolicy.from_vector(values, T_L=policy.T_L))
    return OctoArmPolicy(T_L=policy.T_L, arm_policies=tuple(idle_arms))


def rotate_policy_by_angle(
    policy: OctoArmPolicy, heading_angle: float
) -> OctoArmPolicy:
    arm_vectors = [arm_policy.to_vector() for arm_policy in policy.arm_policies]
    arm_count = len(arm_vectors)
    arm_step = 2.0 * np.pi / arm_count
    shift = float(heading_angle / arm_step)

    rotated_vectors: list[np.ndarray] = []
    for arm_index in range(arm_count):
        source_index = arm_index - shift
        lower_index = int(np.floor(source_index)) % arm_count
        upper_index = (lower_index + 1) % arm_count
        upper_weight = source_index - np.floor(source_index)
        lower_weight = 1.0 - upper_weight
        rotated_vectors.append(
            lower_weight * arm_vectors[lower_index]
            + upper_weight * arm_vectors[upper_index]
        )

    return OctoArmPolicy.from_vector(
        np.concatenate(rotated_vectors, axis=0),
        T_L=policy.T_L,
    )


@njit(cache=True, fastmath=True)
def current_activation(
    phase: float,
    center: float,
    deviation: float,
    scale: float,
) -> float:
    return (
        scale
        * np.exp(-((phase - center) ** 2) / (2 * deviation**2))
        * (1 - (2 * phase - 1) ** 8)
    )
