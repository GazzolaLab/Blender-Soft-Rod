from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import elastica as ea

from virtual_field.core.commands import ArmCommand, MultiArmCommand
from virtual_field.core.state import ArmState, SphereEntity, Transform
from virtual_field.runtime.custom_elastica.dissipation import RayleighDamping
from virtual_field.runtime.foraging_elastica import (
    BaseSphereTether,
    OctoArmPolicy,
    SegmentExtensionActuation,
    SuckerActuation,
    YSurfaceBallwGravity,
    current_activation,
    idle_policy_like,
    rotate_policy_by_angle,
)
from virtual_field.runtime.mode_base import OctoArmSimulationBase
from virtual_field.runtime.orientation import controller_quat_xyzw_to_matrix
from virtual_field.runtime.spirob_elastica.spirob import create_spirob

from importlib.resources import files

# Configurations
ASSET_PATH = files("virtual_field").joinpath("externals", "crawling")
EXTERNAL_POLICY_PATH = ASSET_PATH / "best_policy.npy"


class _Simulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


def _make_head(base) -> ea.CosseratRod:
    start = np.array(base, dtype=np.float64)
    head = ea.CosseratRod.straight_rod(
        n_elements=3,
        start=start,
        direction=np.array([0.0, 1.0, 0.0], dtype=np.float64),
        normal=np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        base_length=0.2,
        base_radius=0.02,
        density=4500.0,
        youngs_modulus=1.0e6,
        position=(
            np.array(
                [
                    [0.0, 0.00, 0.0],
                    [0.0, 0.023, 0.005],
                    [0.0, 0.053, 0.01],
                    [0.0, 0.13, 0.04],
                ],
                dtype=np.float64,
            )
            + start
        ).T,
    )
    head.radius[0] = 0.02
    head.radius[1] = 0.035
    head.radius[2] = 0.08
    head.volume[:] = np.pi * head.radius[:] ** 2 * head.lengths[:]
    head.compute_internal_forces_and_torques(0.0)
    head.rest_sigma[:] = head.sigma
    head.rest_kappa[:] = head.kappa
    return head


def _load_best_policy() -> OctoArmPolicy:
    vector = np.load(EXTERNAL_POLICY_PATH).astype(np.float64)
    return OctoArmPolicy.from_vector(vector, T_L=2.4)


@dataclass(slots=True)
class CathyForagingSimulation(OctoArmSimulationBase):
    _forage_targets: np.ndarray = field(init=False)
    head: ea.CosseratRod = field(init=False)
    head_arm_id: str = field(init=False)
    base_sphere: ea.Sphere = field(init=False)
    base_sphere_radius: float = field(init=False, default=0.09)
    base_range: tuple[int, int] = field(init=False, default=(1, 3))
    middle_range: tuple[int, int] = field(init=False, default=(6, 8))
    base_policy: OctoArmPolicy = field(init=False)
    idle_policy: OctoArmPolicy = field(init=False)
    active_policy: OctoArmPolicy = field(init=False)
    _head_pose: Transform = field(init=False)
    _head_local_positions: np.ndarray = field(init=False)
    _head_local_directors: np.ndarray = field(init=False)
    _crawl_active_until: float = field(init=False, default=0.0)
    _crawl_phase_origin: float = field(init=False, default=0.0)
    _last_crawl_pressed: bool = field(init=False, default=False)
    base_suction_active: np.ndarray = field(init=False)
    middle_suction_active: np.ndarray = field(init=False)
    target_extension: np.ndarray = field(init=False)
    target_stiffness: np.ndarray = field(init=False)
    target_bend: np.ndarray = field(init=False)

    def build_simulation(self) -> None:
        self.simulator = _Simulator()
        self.timestepper = ea.PositionVerlet()

        self.head_arm_id = f"{self.user_id}_head"
        self.head = _make_head(self.base_position)
        self._head_local_positions = np.asarray(
            self.head.position_collection, dtype=np.float64
        ).copy() - np.asarray(self.head.position_collection[:, [0]], dtype=np.float64)
        self._head_local_directors = np.asarray(
            self.head.director_collection, dtype=np.float64
        ).copy()
        self._head_pose = Transform(
            translation=list(self.base_position),
            rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        )

        self.base_policy = _load_best_policy()
        self.idle_policy = idle_policy_like(self.base_policy)
        self.active_policy = self.idle_policy

        arm_count = len(self.arm_ids)
        self.base_suction_active = np.zeros(arm_count, dtype=np.float64)
        self.middle_suction_active = np.zeros(arm_count, dtype=np.float64)
        self.target_extension = np.zeros(arm_count, dtype=np.float64)
        self.target_stiffness = np.ones(arm_count, dtype=np.float64)
        self.target_bend = np.zeros(arm_count, dtype=np.float64)

        self.base_sphere = ea.Sphere(
            np.asarray(self.base_position, dtype=np.float64),
            self.base_sphere_radius,
            100.0,
        )
        self.simulator.append(self.base_sphere)
        self.simulator.add_forcing_to(self.base_sphere).using(
            YSurfaceBallwGravity,
            k_c=1.0e5,
            nu_c=1.0e1,
            plane_origin=-self.base_sphere_radius,
        )
        self.simulator.dampen(self.base_sphere).using(
            RayleighDamping,
            damping_constant=1.0,
            rotational_damping_constant=1.0e-2,
            time_step=self.dt_internal,
        )

        plane_y = -0.02
        rods: dict[str, ea.CosseratRod] = {}
        for arm_index, arm_id in enumerate(self.arm_ids):
            angle = 2.0 * np.pi * (arm_index + 0.5) / arm_count
            direction = np.array(
                [np.sin(angle), 0.0, -np.cos(angle)],
                dtype=np.float64,
            )
            rod = create_spirob(
                15,
                np.asarray(self.base_position, dtype=np.float64),
                direction,
                np.array([0.0, -1.0, 0.0], dtype=np.float64),
                0.45,
                0.02,
                5500.0,
                5.0e5,
            )
            rods[arm_id] = rod
            self.simulator.append(rod)
            self.simulator.detect_contact_between(rod, self.base_sphere).using(
                BaseSphereTether,
                k=1.0e3,
                k_rot=1.0,
                rod_node_index=0,
                relative_rotation=rod.director_collection[..., 0]
                @ self.base_sphere.director_collection[..., 0].T,
            )
            self.simulator.add_forcing_to(rod).using(
                SuckerActuation,
                k=1.0e2,
                nu=5.0e2,
                k_c=3.0,
                nu_c=0.0,
                trigger=lambda idx=arm_index: (
                    float(self.middle_suction_active[idx]),
                    float(self.base_suction_active[idx]),
                ),
                plane_origin=(0.0, plane_y, 0.0),
                plane_normal=(0.0, 1.0, 0.0),
                start_index=[self.middle_range[0], self.base_range[0]],
                end_index=[self.middle_range[1], self.base_range[1]],
                contact_trigger_index=1,
            )
            self.simulator.add_forcing_to(rod).using(
                SegmentExtensionActuation,
                original_shear_matrix=rod.shear_matrix.copy(),
                original_bend_matrix=rod.bend_matrix.copy(),
                start_index=self.base_range[1],
                end_index=self.middle_range[0],
                amplitude=lambda idx=arm_index: (
                    self.target_extension[idx],
                    self.target_stiffness[idx],
                    self.target_bend[idx],
                ),
            )
            self.simulator.dampen(rod).using(
                ea.AnalyticalLinearDamper,
                translational_damping_constant=2.0,
                rotational_damping_constant=3.0e-3,
                time_step=self.dt_internal,
            )

        self.rods = rods
        self.simulator.finalize()

        self._forage_targets = np.array(
            [
                [-0.35, 1.05, -0.55],
                [-0.15, 1.18, -0.80],
                [0.10, 1.10, -0.92],
                [0.34, 1.04, -0.60],
                [0.28, 1.30, -0.35],
                [0.00, 1.26, -0.22],
                [-0.24, 1.22, -0.30],
                [-0.40, 1.14, -0.42],
            ],
            dtype=np.float64,
        )

    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        return

    def handle_frame_command(self, command: MultiArmCommand) -> None:
        if command.head_pose is not None:
            self._head_pose = command.head_pose

        crawl_pressed = bool(command.actions.get("crawl", False))
        if crawl_pressed and not self._last_crawl_pressed:
            heading_angle = self._crawl_heading_angle_from_head_pose()
            self.active_policy = rotate_policy_by_angle(self.base_policy, heading_angle)
            self._crawl_phase_origin = self._time
            self._crawl_active_until = self._time + self.base_policy.T_L
        self._last_crawl_pressed = crawl_pressed

    def _crawl_heading_angle_from_head_pose(self) -> float:
        rotation = controller_quat_xyzw_to_matrix(self._head_pose.rotation_xyzw).T
        forward = rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        horizontal = np.array([forward[0], forward[2]], dtype=np.float64)
        norm = float(np.linalg.norm(horizontal))
        if not np.isfinite(norm) or norm <= 1.0e-9:
            return 0.0
        horizontal /= norm
        return float(np.arctan2(horizontal[0], -horizontal[1]))

    def _apply_policy(self) -> None:
        if self._time < self._crawl_active_until:
            phase = float(
                ((self._time - self._crawl_phase_origin) % self.base_policy.T_L)
                / self.base_policy.T_L
            )
            policy = self.active_policy
        else:
            phase = 0.0
            policy = self.idle_policy

        for arm_index, arm_id in enumerate(self.arm_ids):
            arm_policy = policy.arm_policies[arm_index]
            self.target_stiffness[arm_index] = current_activation(
                phase,
                arm_policy.stiffness_center,
                arm_policy.stiffness_deviation,
                arm_policy.stiffness_scale,
            )
            self.target_extension[arm_index] = current_activation(
                phase,
                arm_policy.extension_center,
                arm_policy.extension_deviation,
                arm_policy.extension_scale,
            ) - current_activation(
                phase,
                arm_policy.contraction_center,
                arm_policy.contraction_deviation,
                arm_policy.contraction_scale,
            )
            self.base_suction_active[arm_index] = current_activation(
                phase,
                arm_policy.base_suction_center,
                arm_policy.base_suction_deviation,
                arm_policy.base_suction_scale,
            )
            self.middle_suction_active[arm_index] = current_activation(
                phase,
                arm_policy.middle_suction_center,
                arm_policy.middle_suction_deviation,
                arm_policy.middle_suction_scale,
            )
            self.target_bend[arm_index] = current_activation(
                phase,
                arm_policy.bend_center,
                arm_policy.bend_deviation,
                arm_policy.bend_scale,
            )

    def _sync_head_pose(self) -> None:
        anchor = np.asarray(
            self.base_sphere.position_collection[:, 0],
            dtype=np.float64,
        ).reshape(3, 1)
        rotation = controller_quat_xyzw_to_matrix(self._head_pose.rotation_xyzw).T
        self.head.position_collection[...] = (
            anchor + rotation @ self._head_local_positions
        )
        rowwise = controller_quat_xyzw_to_matrix(self._head_pose.rotation_xyzw)
        for elem_idx in range(self.head.director_collection.shape[-1]):
            self.head.director_collection[..., elem_idx] = rowwise

    def step(self, dt: float) -> None:
        total = max(0.0, dt)
        if total <= 0.0:
            return
        substeps = max(1, int(np.ceil(total / self.dt_internal)))
        step_dt = total / substeps
        for _ in range(substeps):
            self._apply_policy()
            self._time = self.timestepper.step(self.simulator, self._time, step_dt)
        self._sync_head_pose()

    def arm_states(self) -> dict[str, ArmState]:
        self._sync_head_pose()
        states = OctoArmSimulationBase.arm_states(self)
        states[self.head_arm_id] = self._rod_to_arm_state(self.head_arm_id, self.head)
        return states

    def sphere_entities(self) -> list[SphereEntity]:
        return [
            SphereEntity(
                sphere_id=f"{self.user_id}_forage_target_{index}",
                owner_id=self.user_id,
                translation=position.tolist(),
                radius=0.045,
                color_rgb=[0.42, 0.78, 0.47],
            )
            for index, position in enumerate(self._forage_targets)
        ]
