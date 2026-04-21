from __future__ import annotations

from dataclasses import dataclass, field

import elastica as ea
import numpy as np

from loguru import logger
from numba import njit

from coomm.actuations.muscles import (
    LongitudinalMuscle,
    MuscleGroup,
    ObliqueMuscle,
    TransverseMuscle,
    force_length_weight_poly,
)
from coomm.actuations.actuation import ApplyActuations

from virtual_field.runtime.mode_base import DualArmSimulationBase
from virtual_field.core.state import SphereEntity

# MUSCLE CONFIGURATIONS
# Relative geometry configuration (normalized by base radius)
#LM_RATIO_MUSCLE_POSITION = 0.0075
LM_RATIO_MUSCLE_POSITION = 0.0175
OM_RATIO_MUSCLE_POSITION = 0.01125
AN_RATIO_RADIUS = 0.002
TM_RATIO_RADIUS = 0.045
LM_RATIO_RADIUS = 0.001
OM_RATIO_RADIUS = 0.00075

# Muscle topology and stress parameters
OM_ROTATION_NUMBER = 6
TM_MAX_MUSCLE_STRESS = 15_000.0 * 0
LM_MAX_MUSCLE_STRESS = 10_000.0 * 120 
OM_MAX_MUSCLE_STRESS = 100_000.0 * 4
LM_GROUP_COUNT = 4

# Muscle group ordering for activation assignment
TM_GROUP_INDEX = 0
LM_GROUP_START_INDEX = 1
RIGHT_OM_GROUP_INDEX = 5
LEFT_OM_GROUP_INDEX = 6

# Activation shaping
ACTIVATION_RAMP_TIME = 10

# Reaching controller parameters
SUCKER_EPS = 1.0e-12
SUCKER_A_MAX = 1.0
#SUCKER_GAMMA = 0.0002
SUCKER_GAMMA = 20
SUCKER_PHI = 0.005

damping_constant = 0.1


# TODO: Move these out to the separate file. Keep them here for now, as this mode needs to be tuned isolated.
@njit(cache=True)
def _heaviside_positive(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


@njit(cache=True)
def _cross_cols(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[0, :] = a[1, :] * b[2, :] - a[2, :] * b[1, :]
    out[1, :] = a[2, :] * b[0, :] - a[0, :] * b[2, :]
    out[2, :] = a[0, :] * b[1, :] - a[1, :] * b[0, :]
    return out


@njit(cache=True)
def _safe_normalize_cols(v: np.ndarray, eps: float) -> np.ndarray:
    out = v.copy()
    norms = np.sqrt((out * out).sum(axis=0))
    norms = np.maximum(norms, eps)
    out /= norms[None, :]
    return out


@njit(cache=True)
def _sucker_full_controller_kernel(
    x_c: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    d3: np.ndarray,
    r_xi: np.ndarray,
    target_pos: np.ndarray,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d_xi = d2
    x_center = 0.5 * (x_c[:, :-1] + x_c[:, 1:])
    x_xi = x_center + d_xi * r_xi[None, :]

    rho = target_pos[:, None] - x_xi
    rho_norm = np.sqrt((rho * rho).sum(axis=0))
    rho_norm = np.maximum(rho_norm, eps)
    n = rho / rho_norm[None, :]

    n_center = x_center.shape[1]
    arm_delta = x_c[:, -1] - x_c[:, 0]
    arm_length = np.sqrt((arm_delta * arm_delta).sum())
    s = np.linspace(0.0, arm_length, n_center)
    s_bar = s[np.argmin(rho_norm)]

    
    mu2 = (
        -1.0 + 2.0 / (1.0 + np.exp(-2*SUCKER_GAMMA * s))
    )
    mu = SUCKER_A_MAX * mu2 * (
        1.0 - 1.0 / (1.0 + np.exp(-SUCKER_GAMMA * (s - (s_bar + SUCKER_PHI))))
    )

    proj = (n * d3).sum(axis=0)
    n_t = _safe_normalize_cols(n - d3 * proj[None, :], eps)

    cross_term = _cross_cols(d_xi, n_t)
    sin_alpha = (d3 * cross_term).sum(axis=0)
    cos_alpha = (d_xi * n_t).sum(axis=0)
    alpha_t = np.arctan2(sin_alpha, cos_alpha)

    chi_t = np.sign(np.sin(alpha_t))
    a_om = mu * (np.sin(alpha_t) ** 2)
    a_r_om = a_om * _heaviside_positive(-chi_t)
    a_l_om = a_om * _heaviside_positive(chi_t)

    angles = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0], dtype=np.float64)
    a_lm = np.zeros((LM_GROUP_COUNT, n_center), dtype=np.float64)
    for i in range(LM_GROUP_COUNT):
        theta = angles[i]
        d_i = np.cos(theta) * d1 + np.sin(theta) * d2
        d_bi = _safe_normalize_cols(_cross_cols(d3, d_i), eps)
        proj_b = (n * d_bi).sum(axis=0)
        n_b = _safe_normalize_cols(n - d_bi * proj_b[None, :], eps)

        cross_term_b = _cross_cols(d3, n_b)
        sin_alpha_b = (d_bi * cross_term_b).sum(axis=0)
        cos_alpha_b = (d3 * n_b).sum(axis=0)
        alpha_b_tan = np.arctan2(sin_alpha_b, cos_alpha_b)
        chi_b = np.sign(np.sin(alpha_b_tan))
        a_lm[i, :] = mu * _heaviside_positive(chi_b) * (np.sin(alpha_b_tan) ** 2)

    return a_r_om, a_l_om, a_lm[0], a_lm[1], a_lm[2], a_lm[3]


class ApplyOctopusMuscles(ApplyActuations):
    """ApplyMuscles."""

    def __init__(self, actuations, step_skip: int):
        """__init__.

        Parameters
        ----------
        muscles : Iterable[Muscle]
        step_skip : int
        callback_params_list : list
        """
        super().__init__(actuations, step_skip)
        for m, muscle in enumerate(actuations):
            muscle.index = m


@dataclass(slots=True)
class COOMMOctopusSimulation(DualArmSimulationBase):
    """COOMM Octopus mode"""

    _target_sphere: ea.Sphere = field(init=False)
    _obstacle_sphere: ea.Sphere = field(init=False)
    _muscle_groups: dict[str, list[MuscleGroup]] = field(
        init=False, default_factory=dict
    )
    _activation_buffers: dict[str, list[np.ndarray]] = field(
        init=False, default_factory=dict
    )

    def build_simulation(self) -> None:
        from virtual_field.runtime.spirob_elastica.spirob import create_spirob

        class _Simulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
            ea.Contact,
        ):
            pass

        self.simulator = _Simulator()
        self.timestepper = ea.PositionVerlet()

        n_elem = 21
        # In viewer coordinates, forward is -Z.
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([1.0, 0.0, 0.0])
        base_length = 0.55
        base_radius = 0.01
        density = 2000.0
        youngs_modulus = 1.0e4 * 0.8

        self.left_rod = create_spirob(
            n_elem,
            np.array(self.base_left, dtype=np.float64),
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus,
        )
        self.right_rod = create_spirob(
            n_elem,
            np.array(self.base_right, dtype=np.float64),
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus=youngs_modulus,
        )
        self.simulator.append(self.left_rod)
        self.simulator.append(self.right_rod)

        self._target_sphere = ea.Sphere(
            np.array([0.1, 1.2, -0.3]),
            0.035,
            100.0,
        )
        self.simulator.append(self._target_sphere)

        self._obstacle_sphere = ea.Sphere(
            np.array([-0.0, 1.1, -0.4]),
            0.04,
            100.0,
        )
        self.simulator.append(self._obstacle_sphere)

        self.simulator.detect_contact_between(self.left_rod, self._obstacle_sphere).using(
            ea.RodSphereContact,
            k=2e2,
            nu=0.0,
            velocity_damping_coefficient=1e-6,
            friction_coefficient=1e-6,
        )
        self.simulator.detect_contact_between(self.right_rod, self._obstacle_sphere).using(
            ea.RodSphereContact,
            k=2e2,
            nu=0.0,
            velocity_damping_coefficient=1e-6,
            friction_coefficient=1e-6,
        )

        self.simulator.detect_contact_between(self.left_rod, self._target_sphere).using(
            ea.RodSphereContact,
            k=2e2,
            nu=0.0,
            velocity_damping_coefficient=1e-6,
            friction_coefficient=1e-6,
        )
        self.simulator.detect_contact_between(self.right_rod, self._target_sphere).using(
            ea.RodSphereContact,
            k=2e2,
            nu=0.0,
            velocity_damping_coefficient=1e-6,
            friction_coefficient=1e-6,
        )

        self.simulator.constrain(self.left_rod).using(
            ea.FixedConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )
        self.simulator.constrain(self.right_rod).using(
            ea.FixedConstraint,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        self._muscle_groups[self.arm_ids[0]] = self._create_muscle_groups(
            float(self.left_rod.radius[0]), self.left_rod
        )
        self._muscle_groups[self.arm_ids[1]] = self._create_muscle_groups(
            float(self.right_rod.radius[0]), self.right_rod
        )
        for arm_id in self.arm_ids:
            self._activation_buffers[arm_id] = [
                np.zeros(muscle_group.activation.shape, dtype=np.float64)
                for muscle_group in self._muscle_groups[arm_id]
            ]
        self.simulator.add_forcing_to(self.left_rod).using(
            ApplyOctopusMuscles,
            actuations=self._muscle_groups[self.arm_ids[0]],
            step_skip=1,
        )
        self.simulator.add_forcing_to(self.right_rod).using(
            ApplyOctopusMuscles,
            actuations=self._muscle_groups[self.arm_ids[1]],
            step_skip=1,
        )

        # contact (Comment out for now. We'll consider it later)
        # self.simulator.detect_contact_between(self.left_rod, self.right_rod).using(
        #     ea.RodRodContact, k=1e4, nu=3
        # )
        # self.simulator.detect_contact_between(self.left_rod, self.left_rod).using(
        #     ea.RodSelfContact, k=1e4, nu=3
        # )
        # self.simulator.detect_contact_between(self.right_rod, self.right_rod).using(
        #     ea.RodSelfContact, k=1e4, nu=3
        # )

        #self.simulator.add_forcing_to(self.left_rod).using(
        #    ea.GravityForces,
        #    acc_gravity=np.array([0.0, -9.81, 0.0]),
        #)
        # self.simulator.add_forcing_to(self.right_rod).using(
        #     ea.GravityForces,
        #     acc_gravity=np.array([0.0, -1.00, 0.0]),
        # )

        self.simulator.dampen(self.left_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.010 /5,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.left_rod).using(
            ea.LaplaceDissipationFilter, filter_order=7
        )
        self.simulator.dampen(self.right_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.010 / 5,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.right_rod).using(
            ea.LaplaceDissipationFilter, filter_order=7
        )

        self.simulator.finalize()

    def _create_muscle_groups(
        self, base_radius: float, rod: ea.CosseratRod
    ) -> list[MuscleGroup]:
        lm_ratio_muscle_position = LM_RATIO_MUSCLE_POSITION / base_radius
        om_ratio_muscle_position = OM_RATIO_MUSCLE_POSITION / base_radius
        an_ratio_radius = AN_RATIO_RADIUS / base_radius
        tm_ratio_radius = TM_RATIO_RADIUS / base_radius
        lm_ratio_radius = LM_RATIO_RADIUS / base_radius
        om_ratio_radius = OM_RATIO_RADIUS / base_radius

        rod_area = np.pi * rod.radius**2
        tm_rest_muscle_area = rod_area * (tm_ratio_radius**2 - an_ratio_radius**2)
        lm_rest_muscle_area = rod_area * (lm_ratio_radius**2)
        om_rest_muscle_area = rod_area * (om_ratio_radius**2)

        muscle_dict = dict(force_length_weight=force_length_weight_poly)
        muscle_groups: list[MuscleGroup] = [
            MuscleGroup(
                muscles=[
                    TransverseMuscle(
                        rest_muscle_area=tm_rest_muscle_area,
                        max_muscle_stress=TM_MAX_MUSCLE_STRESS,
                        **muscle_dict,
                    )
                ],
                type_name="TM",
            )
        ]

        for k in range(LM_GROUP_COUNT):
            muscle_groups.append(
                MuscleGroup(
                    muscles=[
                        LongitudinalMuscle(
                            muscle_init_angle=np.pi * 0.5 * k,
                            ratio_muscle_position=lm_ratio_muscle_position,
                            rest_muscle_area=lm_rest_muscle_area,
                            max_muscle_stress=LM_MAX_MUSCLE_STRESS,
                            **muscle_dict,
                        )
                    ],
                    type_name="LM",
                )
            )

        muscle_groups.append(
            MuscleGroup(
                muscles=[
                    ObliqueMuscle(
                        muscle_init_angle=np.pi * 0.5 * m,
                        ratio_muscle_position=om_ratio_muscle_position,
                        rotation_number=OM_ROTATION_NUMBER,
                        rest_muscle_area=om_rest_muscle_area,
                        max_muscle_stress=OM_MAX_MUSCLE_STRESS,
                        **muscle_dict,
                    )
                    for m in range(LM_GROUP_COUNT)
                ],
                type_name="OM",
            )
        )
        muscle_groups.append(
            MuscleGroup(
                muscles=[
                    ObliqueMuscle(
                        muscle_init_angle=np.pi * 0.5 * m,
                        ratio_muscle_position=om_ratio_muscle_position,
                        rotation_number=-OM_ROTATION_NUMBER,
                        rest_muscle_area=om_rest_muscle_area,
                        max_muscle_stress=OM_MAX_MUSCLE_STRESS,
                        **muscle_dict,
                    )
                    for m in range(LM_GROUP_COUNT)
                ],
                type_name="OM",
            )
        )

        for muscle_group in muscle_groups:
            muscle_group.set_current_length_as_rest_length(rod)
        return muscle_groups

    def _sucker_full_controller(
        self, rod: ea.CosseratRod, target_pos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
        a_r_om, a_l_om, a_lm0, a_lm1, a_lm2, a_lm3 = _sucker_full_controller_kernel(
            rod.position_collection,
            rod.director_collection[0, :, :],
            rod.director_collection[1, :, :],
            rod.director_collection[2, :, :],
            rod.radius,
            target_pos,
            SUCKER_EPS,
        )
        return a_r_om, a_l_om, [a_lm0, a_lm1, a_lm2, a_lm3]

    def _controller_activations(
        self, rod: ea.CosseratRod, target_pos: np.ndarray, arm_id: str
    ) -> list[np.ndarray]:
        activations = self._activation_buffers[arm_id]
        for activation in activations:
            activation.fill(0.0)

        a_r, a_l, a_lm = self._sucker_full_controller(rod, target_pos)
        for lm_idx in range(min(LM_GROUP_COUNT, len(a_lm))):
            activations[LM_GROUP_START_INDEX + lm_idx] = a_lm[lm_idx]
        activations[TM_GROUP_INDEX].fill(
            0.0
        )  # keep transverse off (same as journal_reach)
        activations[RIGHT_OM_GROUP_INDEX] = a_r
        activations[LEFT_OM_GROUP_INDEX] = a_l

        weight = min(1.0, self._time / ACTIVATION_RAMP_TIME)
        for i in range(len(activations)):
            activations[i] *= weight
        return activations

    def step(self, dt: float) -> None:
        total = max(0.0, dt)
        if total <= 0.0:
            return
        substeps = max(1, int(np.ceil(total / self.dt_internal)))
        step_dt = total / substeps
        for _ in range(substeps):
        
            t = self._time
            self._target_sphere.position_collection[0, 0] = 0.1 #- 0.15 * np.cos(0.5*t)
            self._target_sphere.position_collection[1, 0] = 1.2 #- 0.15 * np.sin(0.5*t)
            self._target_sphere.position_collection[2, 0] = -0.3         
            # Now read updated position
            target_pos = np.asarray(
                self._target_sphere.position_collection[:, 0], dtype=np.float64
            )
            self._obstacle_sphere.position_collection[0, 0] = 0.0 #- 0.15 * np.cos(0.5*t)
            self._obstacle_sphere.position_collection[1, 0] = 1.1 #- 0.15 * np.sin(0.5*t)
            self._obstacle_sphere.position_collection[2, 0] = -0.4         
            # Now read updated position
            position_obs = np.asarray(
                self._obstacle_sphere.position_collection[:, 0], dtype=np.float64
            )
#        target_pos = np.asarray(
#            self._sphere.position_collection[:, 0], dtype=np.float64
#        )
        for _ in range(substeps):
            for arm_id in self.arm_ids:
                rod = self.rods[arm_id]

                # TODO: Maybe this step can be included within the simulator module.
                activations = self._controller_activations(rod, target_pos, arm_id)


                for muscle_group, activation in zip(
                    self._muscle_groups[arm_id], activations
                ):
                    muscle_group.apply_activation(activation)
            self._time = self.timestepper.step(self.simulator, self._time, step_dt)

            # logger.info(f"hi: {self._time}")

    def sphere_entities(self) -> list[SphereEntity]:
        spheres: list[SphereEntity] = []
        #Target sphere
        position_target = self._target_sphere.position_collection[..., 0]
        spheres.append(
            SphereEntity(
                sphere_id=f"{self.user_id}_coomm_octopus_sphere",
                owner_id=self.user_id,
                translation=position_target.tolist(),
                radius=float(self._target_sphere.radius),
                color_rgb=[0.95, 0.62, 0.32],
            )
        )

        #Obstacle sphere
        position_obs = self._obstacle_sphere.position_collection[..., 0]
        spheres.append(
            SphereEntity(
                sphere_id=f"{self.user_id}_obstacle",
                owner_id=self.user_id,
                translation=position_obs.tolist(),
                radius=float(self._obstacle_sphere.radius),
                color_rgb=[0.9, 0.2, 0.2], #red
            )
        )
        return spheres
