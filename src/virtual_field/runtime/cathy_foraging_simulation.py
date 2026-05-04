from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files

import elastica as ea
import numpy as np

from virtual_field.core.commands import ArmCommand, MultiArmCommand
from virtual_field.core.state import ArmState, SphereEntity, Transform
from virtual_field.runtime.custom_elastica.dissipation import RayleighDamping
from virtual_field.runtime.foraging_elastica import (
    BaseSphereTether,
    OctoArmPolicy,
    SegmentExtensionActuation,
    SuckerActuation,
    YSurfaceBallwGravity,
    YSurfaceRodwGravity,
    current_activation,
    idle_policy_like,
    rotate_policy_by_angle,
)
from virtual_field.runtime.mode_base import OctoArmSimulationBase
from virtual_field.runtime.orientation import controller_quat_xyzw_to_matrix
from virtual_field.runtime.spirob_elastica.spirob import create_spirob

# Configurations
ASSET_PATH = files("virtual_field").joinpath("externals", "crawling")
EXTERNAL_POLICY_PATH = ASSET_PATH / "best_policy.npy"
HEAD_TILT_ACTIVATION = np.sin(np.deg2rad(12.0))
JOYSTICK_ACTIVATION = 0.15
TARGET_SELECTION_TIMEOUT = 6.0
FORAGE_TARGET_RADIUS = 0.045
FORAGE_TARGET_COLOR_RGB = [0.42, 0.78, 0.47]
LEFT_TARGET_COLOR_RGB = [0.96, 0.38, 0.34]
RIGHT_TARGET_COLOR_RGB = [0.34, 0.54, 0.96]
BOTH_TARGET_COLOR_RGB = [0.75, 0.46, 0.94]
SELECTION_DISTANCE_THRESHOLD = 0.09
POINTING_FORWARD = np.array([0.0, -1.0, 0.0], dtype=np.float64)


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
    _right_joystick_command: np.ndarray = field(init=False)
    _tilt_active: bool = field(init=False, default=False)
    _cycle_heading_angle: float = field(init=False, default=0.0)
    _last_cycle_index: int = field(init=False, default=-1)
    _crawl_active_until: float = field(init=False, default=0.0)
    _loco_coast_until: float | None = field(init=False, default=None)
    _last_trigger_pressed: dict[str, bool] = field(init=False, default_factory=dict)
    _selected_target_index_by_hand: dict[str, int | None] = field(
        init=False, default_factory=dict
    )
    _selected_target_position_by_hand: dict[str, list[float] | None] = field(
        init=False, default_factory=dict
    )
    _selected_target_time_by_hand: dict[str, float | None] = field(
        init=False, default_factory=dict
    )
    _selected_arm_id_by_hand: dict[str, str | None] = field(
        init=False, default_factory=dict
    )
    _pull_target_position: dict[str, np.ndarray] = field(
        init=False, default_factory=dict
    )
    _pull_target_orientation: dict[str, np.ndarray] = field(
        init=False, default_factory=dict
    )
    _pull_target_active: dict[str, bool] = field(init=False, default_factory=dict)
    base_suction_active: np.ndarray = field(init=False)
    middle_suction_active: np.ndarray = field(init=False)
    target_extension: np.ndarray = field(init=False)
    target_stiffness: np.ndarray = field(init=False)
    target_bend: np.ndarray = field(init=False)

    def build_simulation(self) -> None:
        # TEMP
        # base_position = self.base_position.copy()
        base_position = np.zeros(3, dtype=np.float64)
        base_position[2] -= 0.5

        from virtual_field.runtime.custom_elastica.control import (
            TargetPoseProportionalControl,
        )

        self.simulator = _Simulator()
        self.timestepper = ea.PositionVerlet()

        self.head_arm_id = f"{self.user_id}_head"
        self.head = _make_head(base_position)
        self._head_local_positions = (
            self.head.position_collection - self.head.position_collection[:, [0]]
        )
        self._head_local_directors = self.head.director_collection.copy()
        self._head_pose = Transform(
            translation=list(base_position),
            rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        self._right_joystick_command = np.zeros(2, dtype=np.float64)

        self.base_policy = _load_best_policy()
        self.idle_policy = idle_policy_like(self.base_policy)
        self.active_policy = self.idle_policy
        self._last_trigger_pressed = {"left": False, "right": False}
        self._selected_target_index_by_hand = {"left": None, "right": None}
        self._selected_target_position_by_hand = {"left": None, "right": None}
        self._selected_target_time_by_hand = {"left": None, "right": None}
        self._selected_arm_id_by_hand = {"left": None, "right": None}

        arm_count = len(self.arm_ids)
        self.base_suction_active = np.zeros(arm_count, dtype=np.float64)
        self.middle_suction_active = np.zeros(arm_count, dtype=np.float64)
        self.target_extension = np.zeros(arm_count, dtype=np.float64)
        self.target_stiffness = np.ones(arm_count, dtype=np.float64)
        self.target_bend = np.zeros(arm_count, dtype=np.float64)

        self.base_sphere = ea.Sphere(
            np.asarray(base_position, dtype=np.float64),
            self.base_sphere_radius,
            100.0,
        )
        self.simulator.append(self.base_sphere)
        self.simulator.add_forcing_to(self.base_sphere).using(
            YSurfaceBallwGravity,
            k_c=1.0e5,
            nu_c=1.0e1,
            plane_origin=-self.base_sphere_radius + base_position[1],
        )
        self.simulator.dampen(self.base_sphere).using(
            RayleighDamping,
            damping_constant=1.0,
            rotational_damping_constant=1.0e-2,
            time_step=self.dt_internal,
        )

        plane_y = -0.02 + base_position[1]
        rods: dict[str, ea.CosseratRod] = {}
        for arm_index, arm_id in enumerate(self.arm_ids):
            angle = 2.0 * np.pi * (arm_index + 0.5) / arm_count
            direction = np.array(
                [np.sin(angle), 0.0, -np.cos(angle)],
                dtype=np.float64,
            )
            rod = create_spirob(
                15,
                np.asarray(base_position, dtype=np.float64),
                direction,
                np.array([0.0, -1.0, 0.0], dtype=np.float64),
                0.45,
                0.02,
                5500.0,
                5.0e5,
            )
            rods[arm_id] = rod
            self.simulator.append(rod)
            self.simulator.add_forcing_to(rod).using(
                YSurfaceRodwGravity,
                k_c=1.0e5,
                nu_c=1.0e1,
                plane_origin=plane_y + base_position[1],
            )
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
            self.simulator.add_forcing_to(rod).using(
                TargetPoseProportionalControl,
                elem_index=-1,
                p_linear_value=50.0,
                p_angular_value=0.0,
                target=lambda arm_id=arm_id: self._pull_target_for_arm(arm_id),
                is_attached=lambda arm_id=arm_id: self._pull_target_active[arm_id],
                ramp_up_time=1e-3,
            )
            self.simulator.dampen(rod).using(
                ea.AnalyticalLinearDamper,
                translational_damping_constant=2.0,
                rotational_damping_constant=3.0e-3,
                time_step=self.dt_internal,
            )

        self.rods = rods
        self._pull_target_position = {
            arm_id: rod.position_collection[:, -1].copy()
            for arm_id, rod in rods.items()
        }
        self._pull_target_orientation = {
            arm_id: rod.director_collection[..., -1].copy()
            for arm_id, rod in rods.items()
        }
        self._pull_target_active = {arm_id: False for arm_id in self.arm_ids}
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
        self._forage_targets += np.array([0.0, -0.9, -0.5], dtype=np.float64)

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
        right_controller_command = command.commands.get(self.arm_ids[1])
        if right_controller_command is None:
            self._right_joystick_command.fill(0.0)
        else:
            self._right_joystick_command[0] = right_controller_command.joystick[0]
            self._right_joystick_command[1] = right_controller_command.joystick[1]
        self._update_target_selection_from_command(command)

    def _update_target_selection_from_command(self, command: MultiArmCommand) -> None:
        controller_arm_ids = {
            "left": self.arm_ids[0],
            "right": self.arm_ids[1],
        }
        for hand, arm_id in controller_arm_ids.items():
            controller_command = command.commands.get(arm_id)
            if controller_command is None:
                self._last_trigger_pressed[hand] = False
                continue
            pressed = bool(controller_command.buttons.get("trigger_click", False))
            was_pressed = self._last_trigger_pressed.get(hand, False)
            if pressed and not was_pressed:
                target_index = self._selected_target_index_from_transform(
                    controller_command.target
                )
                if target_index is not None:
                    if self._selected_target_index_by_hand[hand] == target_index:
                        self._clear_selected_target(hand)
                    else:
                        self._selected_target_index_by_hand[hand] = target_index
                        self._selected_target_position_by_hand[hand] = (
                            self._forage_targets[target_index].tolist()
                        )
                        self._selected_target_time_by_hand[hand] = self._time
                        self._selected_arm_id_by_hand[hand] = (
                            self._closest_arm_id_to_target(
                                self._forage_targets[target_index]
                            )
                        )
            self._last_trigger_pressed[hand] = pressed
        self._refresh_pull_targets()

    def _selected_target_index_from_transform(self, transform: Transform) -> int | None:
        origin = np.asarray(transform.translation, dtype=np.float64)
        rotation = controller_quat_xyzw_to_matrix(transform.rotation_xyzw).T
        direction = rotation @ POINTING_FORWARD
        direction_norm = float(np.linalg.norm(direction))
        if (not np.isfinite(direction_norm)) or direction_norm <= 1.0e-9:
            return None
        direction /= direction_norm

        best_index: int | None = None
        best_distance = np.inf
        best_t = np.inf
        for index, target in enumerate(self._forage_targets):
            offset = target - origin
            t = float(np.dot(offset, direction))
            if (not np.isfinite(t)) or t < 0.0:
                continue
            closest = origin + t * direction
            distance = float(np.linalg.norm(target - closest))
            if distance <= SELECTION_DISTANCE_THRESHOLD and (
                distance < best_distance or (distance == best_distance and t < best_t)
            ):
                best_index = index
                best_distance = distance
                best_t = t
        return best_index

    def selected_target_indices(self) -> dict[str, int | None]:
        return dict(self._selected_target_index_by_hand)

    def selected_target_positions(self) -> dict[str, list[float] | None]:
        return {
            hand: None if position is None else list(position)
            for hand, position in self._selected_target_position_by_hand.items()
        }

    def _closest_arm_id_to_target(self, target: np.ndarray) -> str:
        base_center = np.asarray(
            self.base_sphere.position_collection[:, 0], dtype=np.float64
        )
        target_offset = np.asarray(target, dtype=np.float64) - base_center
        target_horizontal_sq = (
            target_offset[0] * target_offset[0] + target_offset[2] * target_offset[2]
        )
        if target_horizontal_sq <= 1.0e-12:
            return self.arm_ids[0]

        target_angle = float(np.arctan2(target_offset[0], -target_offset[2]))
        closest_arm_id = self.arm_ids[0]
        closest_delta = np.inf
        for arm_id in self.arm_ids:
            rod = self.rods[arm_id]
            # Use the arm's local radial heading (base segment direction), not
            # the base node position, since all rods are anchored near the same
            # center point.
            radial = np.asarray(
                rod.position_collection[:, 1], dtype=np.float64
            ) - np.asarray(rod.position_collection[:, 0], dtype=np.float64)
            radial_horizontal_sq = radial[0] * radial[0] + radial[2] * radial[2]
            if radial_horizontal_sq <= 1.0e-12:
                continue
            anchor_angle = float(np.arctan2(radial[0], -radial[2]))
            delta = float(
                np.abs(
                    np.arctan2(
                        np.sin(target_angle - anchor_angle),
                        np.cos(target_angle - anchor_angle),
                    )
                )
            )
            if delta < closest_delta:
                closest_delta = delta
                closest_arm_id = arm_id
        return closest_arm_id

    def _pull_target_for_arm(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        return self._pull_target_position[arm_id], self._pull_target_orientation[arm_id]

    def _clear_selected_target(self, hand: str) -> None:
        self._selected_target_index_by_hand[hand] = None
        self._selected_target_position_by_hand[hand] = None
        self._selected_target_time_by_hand[hand] = None
        self._selected_arm_id_by_hand[hand] = None

    def _release_completed_or_expired_targets(self) -> None:
        for hand in ("left", "right"):
            selected_time = self._selected_target_time_by_hand[hand]
            selected_position = self._selected_target_position_by_hand[hand]
            selected_arm_id = self._selected_arm_id_by_hand[hand]
            if (
                selected_time is None
                or selected_position is None
                or selected_arm_id is None
            ):
                continue

            if self._time - selected_time >= TARGET_SELECTION_TIMEOUT:
                self._clear_selected_target(hand)
                continue

            rod = self.rods[selected_arm_id]
            tip_position = rod.position_collection[:, -1]
            target_position = np.asarray(selected_position, dtype=np.float64)
            touch_distance = FORAGE_TARGET_RADIUS + float(rod.radius[-1])
            if float(np.linalg.norm(tip_position - target_position)) <= touch_distance:
                self._clear_selected_target(hand)

        self._refresh_pull_targets()

    def _refresh_pull_targets(self) -> None:
        for arm_id in self.arm_ids:
            self._pull_target_active[arm_id] = False

        for hand in ("left", "right"):
            arm_id = self._selected_arm_id_by_hand[hand]
            position = self._selected_target_position_by_hand[hand]
            if arm_id is None or position is None:
                continue

            target_position = np.asarray(position, dtype=np.float64)
            rod = self.rods[arm_id]
            self._pull_target_position[arm_id] = target_position
            self._pull_target_orientation[arm_id] = rod.director_collection[
                ..., -1
            ].copy()
            self._pull_target_active[arm_id] = True

    def _right_joystick_heading(self) -> tuple[bool, float]:
        command_x = float(self._right_joystick_command[0])
        command_y = float(-self._right_joystick_command[1])
        joystick_norm = float(np.hypot(command_x, command_y))
        if (not np.isfinite(joystick_norm)) or joystick_norm < JOYSTICK_ACTIVATION:
            return False, self._cycle_heading_angle

        rotation = controller_quat_xyzw_to_matrix(self._head_pose.rotation_xyzw).T
        forward = rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        forward_horizontal = np.array([forward[0], forward[2]], dtype=np.float64)
        forward_norm = float(np.linalg.norm(forward_horizontal))
        if (not np.isfinite(forward_norm)) or forward_norm <= 1.0e-9:
            return False, self._cycle_heading_angle

        forward_horizontal /= forward_norm
        right_horizontal = np.array(
            [-forward_horizontal[1], forward_horizontal[0]],
            dtype=np.float64,
        )
        horizontal = command_y * forward_horizontal + command_x * right_horizontal
        horizontal_norm = float(np.linalg.norm(horizontal))
        if (not np.isfinite(horizontal_norm)) or horizontal_norm <= 1.0e-9:
            return False, self._cycle_heading_angle
        horizontal /= horizontal_norm
        return True, float(np.arctan2(horizontal[0], -horizontal[1]))

    # Head-tilt steering is intentionally disabled for now while we evaluate
    # joystick steering in VR. Keeping the previous logic here makes it easy to
    # revive later without re-deriving the thresholds and heading math.
    #
    # def _head_tilt_heading(self) -> tuple[bool, float]:
    #     rotation = controller_quat_xyzw_to_matrix(self._head_pose.rotation_xyzw).T
    #     forward = rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    #     horizontal = np.array([forward[0], forward[2]], dtype=np.float64)
    #     up = rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float64)
    #     tilt_amount = float(
    #         np.linalg.norm(np.array([up[0], up[2]], dtype=np.float64))
    #     )
    #     norm = float(np.linalg.norm(horizontal))
    #     if (
    #         (not np.isfinite(tilt_amount))
    #         or tilt_amount < HEAD_TILT_ACTIVATION
    #         or (not np.isfinite(norm))
    #         or norm <= 1.0e-9
    #     ):
    #         return False, self._cycle_heading_angle
    #     horizontal /= norm
    #     return True, float(np.arctan2(horizontal[0], -horizontal[1]))

    def _sync_coast_deadline(
        self, was_crawling: bool, has_target: bool, period: float
    ) -> None:
        self._tilt_active = has_target
        if has_target:
            self._loco_coast_until = None
            return
        if was_crawling:
            self._loco_coast_until = float(
                (np.floor(self._time / period) + 1.0) * period
            )
        elif (
            self._loco_coast_until is not None and self._time >= self._loco_coast_until
        ):
            self._loco_coast_until = None

    def _locomotion_is_active(self) -> bool:
        return self._tilt_active or (
            self._loco_coast_until is not None and self._time < self._loco_coast_until
        )

    def _set_active_policy_heading(
        self, heading_angle: float, cycle_index: int
    ) -> None:
        self._cycle_heading_angle = heading_angle
        self.active_policy = rotate_policy_by_angle(self.base_policy, heading_angle)
        self._last_cycle_index = cycle_index

    def _refresh_heading_and_active_policy(
        self, heading_angle: float, period: float
    ) -> None:
        cycle_index = int(np.floor(self._time / period))
        self._crawl_active_until = (cycle_index + 1) * period
        if (
            cycle_index != self._last_cycle_index
            or self.active_policy is self.idle_policy
        ):
            self._set_active_policy_heading(heading_angle, cycle_index)

    def _write_tentacle_actuation(self, phase: float, policy: OctoArmPolicy) -> None:
        for arm_index, _arm_id in enumerate(self.arm_ids):
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

    def _apply_policy(self) -> None:
        was_crawling = self._tilt_active
        period = float(self.base_policy.T_L)
        has_target, heading_angle = self._right_joystick_heading()
        # has_target, heading_angle = self._head_tilt_heading()
        self._sync_coast_deadline(was_crawling, has_target, period)

        if self._locomotion_is_active():
            phase = float((self._time % period) / period)
            if has_target:
                self._refresh_heading_and_active_policy(heading_angle, period)
            policy = self.active_policy
        else:
            phase = 0.0
            policy = self.idle_policy
            self._last_cycle_index = -1

        self._write_tentacle_actuation(phase, policy)

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
        self._release_completed_or_expired_targets()
        self._sync_head_pose()

    def arm_states(self) -> dict[str, ArmState]:
        self._sync_head_pose()
        states = OctoArmSimulationBase.arm_states(self)
        states[self.head_arm_id] = self._rod_to_arm_state(self.head_arm_id, self.head)
        return states

    def sphere_entities(self) -> list[SphereEntity]:
        left_index = self._selected_target_index_by_hand["left"]
        right_index = self._selected_target_index_by_hand["right"]
        return [
            SphereEntity(
                sphere_id=f"{self.user_id}_forage_target_{index}",
                owner_id=self.user_id,
                translation=position.tolist(),
                radius=FORAGE_TARGET_RADIUS,
                color_rgb=(
                    BOTH_TARGET_COLOR_RGB
                    if left_index == index and right_index == index
                    else (
                        LEFT_TARGET_COLOR_RGB
                        if left_index == index
                        else (
                            RIGHT_TARGET_COLOR_RGB
                            if right_index == index
                            else FORAGE_TARGET_COLOR_RGB
                        )
                    )
                ),
            )
            for index, position in enumerate(self._forage_targets)
        ]
