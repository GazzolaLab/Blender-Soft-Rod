from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

from dataclasses import dataclass, field
import numpy as np

from virtual_field.core.state import ArmState, MeshEntity, SphereEntity, Transform
from virtual_field.core.commands import ArmCommand
from virtual_field.runtime.orientation import (
    compose_rowwise_directors,
    controller_quat_xyzw_to_matrix,
    invert_rowwise_director,
    matrix_to_quat_xyzw,
)

class SimulationBase(ABC):
    pass

@dataclass(slots=True)
class DualArmSimulationBase(SimulationBase, ABC):
    """Common control/state helpers shared by dual-arm runtime modes."""

    # Parameters
    user_id: str
    arm_ids: tuple[str, str]
    base_left: list[float]
    base_right: list[float]

    # State
    simulator: Any = field(init=False)
    timestepper: Any = field(init=False)
    left_rod: Any = field(init=False)
    right_rod: Any = field(init=False)
    dt_internal: float = 1.0e-4
    _time: float = field(init=False, default=0.0)
    _last_log_time: float = field(init=False, default=0.0)
    _target_position: dict[str, np.ndarray] = field(init=False)
    _target_orientation: dict[str, np.ndarray] = field(init=False)
    _rest_target_position: dict[str, np.ndarray] = field(init=False)
    _rest_target_orientation: dict[str, np.ndarray] = field(init=False)
    _base_orientation: dict[str, np.ndarray] = field(init=False)
    _controller_orientation_offset: dict[str, np.ndarray] = field(init=False)
    _attached: dict[str, bool] = field(init=False)

    @final
    def __post_init__(self) -> None:
        self.build_simulation()
        self._initialize_dual_arm_targets()
        self.post_mode_setup()

    @abstractmethod
    def build_simulation(self) -> None:
        """Build the elastica simulation and initialize simulator and rods."""

    def post_mode_setup(self) -> None:
        """Optional hook for advanced subclasses after base target setup."""
        return None

    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        """
        Handle controller commands for a given arm.
        Override this method to customize the controller command mapping.
        """
        primary_pressed = bool(controller_command.buttons.get("primary", False))
        secondary_pressed = bool(controller_command.buttons.get("secondary", False))
        previous_secondary_pressed = bool(
            previous_controller_command
            and previous_controller_command.buttons.get("secondary", False)
        )

        self.set_attached(arm_id, not (primary_pressed or secondary_pressed))
        if secondary_pressed and not previous_secondary_pressed:
            self.reset_target_to_rest(arm_id)
            self.recalibrate_orientation_to_base(
                arm_id, controller_command.target.rotation_xyzw
            )
            return

        self.set_target_pose(
            arm_id=arm_id,
            translation=controller_command.target.translation,
            rotation_xyzw=controller_command.target.rotation_xyzw,
        )

    def handle_command_inactive(self, arm_id: str) -> None:
        """
        Reset per-frame controller-driven state for an arm when no command arrives.
        """
        self.set_attached(arm_id, True)

    def _initialize_dual_arm_targets(self) -> None:
        left_tip_position = self.left_rod.position_collection[:, -1].copy()
        right_tip_position = self.right_rod.position_collection[:, -1].copy()
        left_tip_orientation = self.left_rod.director_collection[..., -1].copy()
        right_tip_orientation = self.right_rod.director_collection[..., -1].copy()

        self._target_position = {
            self.arm_ids[0]: left_tip_position.copy(),
            self.arm_ids[1]: right_tip_position.copy(),
        }
        self._target_orientation = {
            self.arm_ids[0]: left_tip_orientation.copy(),
            self.arm_ids[1]: right_tip_orientation.copy(),
        }
        self._rest_target_position = {
            self.arm_ids[0]: left_tip_position.copy(),
            self.arm_ids[1]: right_tip_position.copy(),
        }
        self._rest_target_orientation = {
            self.arm_ids[0]: left_tip_orientation.copy(),
            self.arm_ids[1]: right_tip_orientation.copy(),
        }
        self._base_orientation = {
            self.arm_ids[0]: self.left_rod.director_collection[..., 0].copy(),
            self.arm_ids[1]: self.right_rod.director_collection[..., 0].copy(),
        }
        self._controller_orientation_offset = {
            self.arm_ids[0]: np.eye(3, dtype=np.float64),
            self.arm_ids[1]: np.eye(3, dtype=np.float64),
        }
        self._attached = {
            self.arm_ids[0]: True,
            self.arm_ids[1]: True,
        }

    def set_target_pose(
        self, arm_id: str, translation: list[float], rotation_xyzw: list[float]
    ) -> None:
        if arm_id not in self._target_position:
            return
        self._target_position[arm_id] = np.array(translation, dtype=np.float64)
        controller_orientation = controller_quat_xyzw_to_matrix(rotation_xyzw)
        self._target_orientation[arm_id] = compose_rowwise_directors(
            self._controller_orientation_offset[arm_id], controller_orientation
        )

    def reset_target_to_rest(self, arm_id: str) -> None:
        if arm_id not in self._rest_target_position:
            return
        self._target_position[arm_id] = self._rest_target_position[arm_id].copy()
        self._target_orientation[arm_id] = self._rest_target_orientation[arm_id].copy()

    def recalibrate_orientation_to_base(
        self, arm_id: str, controller_rotation_xyzw: list[float]
    ) -> None:
        if arm_id not in self._controller_orientation_offset:
            return
        controller_orientation = controller_quat_xyzw_to_matrix(controller_rotation_xyzw)
        self._controller_orientation_offset[arm_id] = compose_rowwise_directors(
            self._base_orientation[arm_id],
            invert_rowwise_director(controller_orientation),
        )

    # Controller Query Methods
    def _target_for_arm(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._target_position[arm_id],
            self._target_orientation[arm_id],
        )

    def get_target_left(self) -> tuple[np.ndarray, np.ndarray]:
        return self._target_for_arm(self.arm_ids[0])

    def get_target_right(self) -> tuple[np.ndarray, np.ndarray]:
        return self._target_for_arm(self.arm_ids[1])

    def _is_arm_attached(self, arm_id: str) -> bool:
        return self._attached[arm_id]

    def is_left_attached(self) -> bool:
        return self._is_arm_attached(self.arm_ids[0])

    def is_right_attached(self) -> bool:
        return self._is_arm_attached(self.arm_ids[1])

    # Controller-driven actions
    def set_attached(self, arm_id: str, attached: bool) -> None:
        if arm_id not in self._attached:
            return
        self._attached[arm_id] = attached

    # Simulation Stepping
    def step(self, dt: float) -> None:
        total = max(0.0, dt)
        if total <= 0.0:
            return
        substeps = max(1, int(np.ceil(total / self.dt_internal)))
        step_dt = total / substeps
        for _ in range(substeps):
            self._time = self.timestepper.step(self.simulator, self._time, step_dt)
        if self._time - self._last_log_time >= 0.1:
            self._last_log_time = self._time

    # Asset Publishing
    def mesh_entities(self) -> list[MeshEntity]:
        return []

    def sphere_entities(self) -> list[SphereEntity]:
        return []

    def arm_states(self) -> dict[str, ArmState]:
        return {
            self.arm_ids[0]: self._rod_to_arm_state(self.arm_ids[0], self.left_rod),
            self.arm_ids[1]: self._rod_to_arm_state(self.arm_ids[1], self.right_rod),
        }

    def _rod_to_arm_state(self, arm_id: str, rod: object) -> ArmState:
        positions = np.asarray(rod.position_collection, dtype=np.float64)
        centerline = positions.T.tolist()
        radii = np.asarray(rod.radius, dtype=np.float64).tolist()
        element_lengths = self._rod_element_lengths(rod)
        directors = np.asarray(rod.director_collection, dtype=np.float64)
        directors_list = [
            directors[..., elem_idx].tolist() for elem_idx in range(directors.shape[-1])
        ]

        base_position = positions[:, 0]
        tip_position = positions[:, -1]
        tip_orientation = rod.director_collection[..., -1]

        return ArmState(
            arm_id=arm_id,
            owner_user_id=self.user_id,
            base=Transform(
                translation=base_position.tolist(),
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            tip=Transform(
                translation=tip_position.tolist(),
                rotation_xyzw=matrix_to_quat_xyzw(tip_orientation.T),
            ),
            centerline=centerline,
            radii=radii,
            element_lengths=element_lengths,
            directors=directors_list,
            contact_points=self.contact_points_for_arm(arm_id),
        )

    def _rod_element_lengths(self, rod: object) -> list[float]:
        lengths = getattr(rod, "lengths", None)
        if lengths is None:
            return []
        return np.asarray(lengths, dtype=np.float64).tolist()

    def contact_points_for_arm(self, arm_id: str) -> list[list[float]]:
        return []
