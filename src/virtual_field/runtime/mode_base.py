from __future__ import annotations

from typing import Any, Protocol, final

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import cos, pi, sin

import numpy as np
from loguru import logger

from virtual_field.core.commands import ArmCommand, MultiArmCommand
from virtual_field.core.state import (
    ArmState,
    HapticEvent,
    MeshEntity,
    SphereEntity,
    Transform,
)
from virtual_field.runtime.orientation import (
    compose_rowwise_directors,
    controller_quat_xyzw_to_matrix,
    invert_rowwise_director,
    matrix_to_quat_xyzw,
)


# Simplest rod protocol.
class Rod(Protocol):
    position_collection: np.ndarray
    director_collection: np.ndarray
    radius: np.ndarray
    lengths: np.ndarray


@dataclass(slots=True, kw_only=True)
class SimulationBase(ABC):
    """Shared backend contract and rod-state helpers for simulation modes."""

    user_id: str
    arm_ids: tuple[str, ...]
    arm_bases: dict[str, list[float]] = field(init=False)
    simulator: Any = field(init=False)
    timestepper: Any = field(init=False)
    rods: dict[str, Any] = field(init=False, default_factory=dict)
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
        self.configure_arm_bases()
        self.build_simulation()
        self._initialize_arm_targets()
        self.post_mode_setup()

    # --- Abstract / Override following methods ---

    @abstractmethod
    def configure_arm_bases(self) -> None:
        """Populate ``self.arm_bases`` before building the simulation."""

    @abstractmethod
    def build_simulation(self) -> None:
        """Build the simulation and populate ``self.rods``."""

    def post_mode_setup(self) -> None:
        pass

    def handle_frame_command(self, command: MultiArmCommand) -> None:
        pass

    @abstractmethod
    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        """Handle controller input for a single arm.

        Subclasses are expected to interpret controller buttons and target pose
        updates in a mode-appropriate way.
        """
        raise NotImplementedError

    def handle_command_inactive(self, arm_id: str) -> None:
        self.set_attached(arm_id, True)

    # ---

    def _initialize_arm_targets(self) -> None:
        self._target_position = {}
        self._target_orientation = {}
        self._rest_target_position = {}
        self._rest_target_orientation = {}
        self._base_orientation = {}
        self._controller_orientation_offset = {}
        self._attached = {}

        for arm_id in self.arm_ids:
            rod = self.rods[arm_id]
            tip_position = np.asarray(
                rod.position_collection[:, -1], dtype=np.float64
            ).copy()
            tip_orientation = np.asarray(
                rod.director_collection[..., -1], dtype=np.float64
            ).copy()
            base_orientation = np.asarray(
                rod.director_collection[..., 0], dtype=np.float64
            ).copy()

            self._target_position[arm_id] = tip_position.copy()
            self._target_orientation[arm_id] = tip_orientation.copy()
            self._rest_target_position[arm_id] = tip_position.copy()
            self._rest_target_orientation[arm_id] = tip_orientation.copy()
            self._base_orientation[arm_id] = base_orientation
            self._controller_orientation_offset[arm_id] = np.array(
                [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64
            )  # Arm-forward quaternion to row-wise orientation
            self._attached[arm_id] = True

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
        self._target_position[arm_id] = self._rest_target_position[
            arm_id
        ].copy()
        self._target_orientation[arm_id] = self._rest_target_orientation[
            arm_id
        ].copy()

    def recalibrate_orientation_to_base(
        self, arm_id: str, controller_rotation_xyzw: list[float]
    ) -> None:
        if arm_id not in self._controller_orientation_offset:
            return
        controller_orientation = controller_quat_xyzw_to_matrix(
            controller_rotation_xyzw
        )
        tt = compose_rowwise_directors(
            self._base_orientation[arm_id],
            invert_rowwise_director(controller_orientation),
        )
        logger.debug(
            "Recalibrated arm {} orientation offset={}",
            arm_id,
            np.round(tt, decimals=6).tolist(),
        )
        self._controller_orientation_offset[arm_id] = tt

    def target_for_arm(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        return (
            self._target_position[arm_id],
            self._target_orientation[arm_id],
        )

    def is_arm_attached(self, arm_id: str) -> bool:
        return self._attached[arm_id]

    def set_attached(self, arm_id: str, attached: bool) -> None:
        if arm_id not in self._attached:
            return
        self._attached[arm_id] = attached

    def step(self, dt: float) -> None:
        total = max(0.0, dt)
        if total <= 0.0:
            return
        substeps = max(1, int(np.ceil(total / self.dt_internal)))
        step_dt = total / substeps
        for _ in range(substeps):
            self._time = self.timestepper.step(
                self.simulator, self._time, step_dt
            )
        if self._time - self._last_log_time >= 0.1:
            self._last_log_time = self._time

    def arm_states(self) -> dict[str, ArmState]:
        return {
            arm_id: self._rod_to_arm_state(arm_id, self.rods[arm_id])
            for arm_id in self.arm_ids
        }

    def mesh_entities(self) -> list[MeshEntity]:
        return []

    def sphere_entities(self) -> list[SphereEntity]:
        return []

    def haptic_events(self) -> list[HapticEvent]:
        return []

    def _rod_to_arm_state(self, arm_id: str, rod: Rod) -> ArmState:
        """
        Convert a rod to an arm state.
        Used to publish the arm state to the client.
        """
        node_slice = self._rod_slice_or_full(rod, "_active_node_slice")
        elem_slice = self._rod_slice_or_full(rod, "_active_elem_slice")
        positions = np.asarray(
            rod.position_collection[:, node_slice], dtype=np.float64
        )
        centerline = positions.T.tolist()
        radii = np.asarray(rod.radius[elem_slice], dtype=np.float64).tolist()
        element_lengths = np.asarray(
            rod.lengths[elem_slice], dtype=np.float64
        ).tolist()
        directors = np.asarray(
            rod.director_collection[:, :, elem_slice], dtype=np.float64
        )
        directors_list = [
            directors[..., elem_idx].tolist()
            for elem_idx in range(directors.shape[-1])
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

    def contact_points_for_arm(self, arm_id: str) -> list[list[float]]:
        return []

    def _rod_slice_or_full(self, rod: Rod, method_name: str) -> slice:
        method = getattr(rod, method_name, None)
        if callable(method):
            return method()
        return slice(None)


@dataclass(slots=True, kw_only=True)
class DualArmSimulationBase(SimulationBase, ABC):
    """Simulation base specialized for a left/right pair of arms."""

    base_left: list[float]
    base_right: list[float]

    def configure_arm_bases(self) -> None:
        self.arm_bases = {
            self.arm_ids[0]: self.base_left,
            self.arm_ids[1]: self.base_right,
        }

    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        primary_pressed = bool(controller_command.buttons.get("primary", False))
        secondary_pressed = bool(
            controller_command.buttons.get("secondary", False)
        )
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

    @property
    def left_rod(self) -> Any:
        return self.rods[self.arm_ids[0]]

    @left_rod.setter
    def left_rod(self, rod: Any) -> None:
        self.rods[self.arm_ids[0]] = rod

    @property
    def right_rod(self) -> Any:
        return self.rods[self.arm_ids[1]]

    @right_rod.setter
    def right_rod(self, rod: Any) -> None:
        self.rods[self.arm_ids[1]] = rod

    def get_target_left(self) -> tuple[np.ndarray, np.ndarray]:
        return self.target_for_arm(self.arm_ids[0])

    def get_target_right(self) -> tuple[np.ndarray, np.ndarray]:
        return self.target_for_arm(self.arm_ids[1])

    def is_left_attached(self) -> bool:
        return self.is_arm_attached(self.arm_ids[0])

    def is_right_attached(self) -> bool:
        return self.is_arm_attached(self.arm_ids[1])


@dataclass(slots=True, kw_only=True)
class OctoArmSimulationBase(SimulationBase, ABC):
    """Simulation base specialized for a fixed eight-arm layout."""

    base_position: tuple[float, float, float]

    arm_radial_spacing: float = 0.05

    def configure_arm_bases(self) -> None:
        base_x, base_y, base_z = self.base_position

        self.arm_bases = {}
        offset_angle = np.deg2rad(22.5)
        for index, arm_id in enumerate(self.arm_ids):
            loc = index / 8.0
            translation = (
                base_x
                - self.arm_radial_spacing * cos(offset_angle + 2.0 * pi * loc),
                base_y,
                base_z
                + self.arm_radial_spacing * sin(offset_angle + 2.0 * pi * loc),
            )
            self.arm_bases[arm_id] = translation
