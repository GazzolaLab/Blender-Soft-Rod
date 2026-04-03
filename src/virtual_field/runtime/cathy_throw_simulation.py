from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from collections import deque

from virtual_field.core.commands import ArmCommand
from virtual_field.core.state import SphereEntity
from virtual_field.runtime.mode_base import DualArmSimulationBase
from .custom_elastica.dissipation import RayleighDamping
from .custom_elastica.contacts import SuckerActuation
from .custom_elastica.forcing import _PullSphereToPoint, _SphereBoxed


@dataclass(slots=True)
class CathyThrowSimulation(DualArmSimulationBase):
    contact_point_history_seconds: float = 1.0
    max_contact_points_visible: int = 2000
    max_contact_points_memory: int = 8000
    _recording_queues: dict[str, deque[tuple[float, list[float]]]] = field(init=False)
    spheres: list[Any] = field(init=False)
    _sucker_active: dict[str, bool] = field(init=False)
    _base_pull_active: dict[str, bool] = field(init=False)

    def build_simulation(self) -> None:
        if self.contact_point_history_seconds <= 0.0:
            raise ValueError("contact_point_history_seconds must be > 0")
        if self.max_contact_points_visible <= 0:
            raise ValueError("max_contact_points_visible must be > 0")
        if self.max_contact_points_memory <= 0:
            raise ValueError("max_contact_points_memory must be > 0")

        import elastica as ea
        from virtual_field.runtime.spirob_elastica.constraints import (
            _SpirobBendConstraint,
        )
        from virtual_field.runtime.spirob_elastica.spirob import create_spirob
        from virtual_field.runtime.custom_elastica.control import (
            TargetPoseProportionalControl,
        )


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
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([1.0, 0.0, 0.0])
        base_length = 0.551479602
        base_radius = 0.019945905
        density = 2500.0
        youngs_modulus = 5.0e5

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
            -normal,
            base_length,
            base_radius,
            density,
            youngs_modulus,
        )
        self.simulator.append(self.left_rod)
        self.simulator.append(self.right_rod)

        self._sucker_active = {
            self.arm_ids[0]: False,
            self.arm_ids[1]: False,
        }
        self._base_pull_active = {
            self.arm_ids[0]: False,
            self.arm_ids[1]: False,
        }

        sphere_box = np.array(
            [
                [-0.65, 0.55, -1.05],
                [0.65, 1.55, 0.05],
            ],
            dtype=np.float64,
        )
        sphere_center = np.array([0.0, 1.10, -0.55], dtype=np.float64)
        sphere_radius = 0.08
        sphere_density = 300.0
        n_sphere = 1
        self.spheres = []
        for _ in range(n_sphere):
            sphere = ea.Sphere(sphere_center.copy(), sphere_radius, sphere_density)
            self.spheres.append(sphere)
            self.simulator.append(sphere)
            self.simulator.add_forcing_to(sphere).using(
                _SphereBoxed,
                bounding_box=sphere_box,
            )
            self.simulator.detect_contact_between(self.left_rod, sphere).using(
                ea.RodSphereContact, k=1e4, nu=0.0
            )
            self.simulator.detect_contact_between(self.right_rod, sphere).using(
                ea.RodSphereContact, k=1e4, nu=0.0
            )
            damping_constant = 1e0
            self.simulator.dampen(sphere).using(
                RayleighDamping,
                damping_constant=damping_constant,
                rotational_damping_constant=damping_constant * 1.0,
                time_step=self.dt_internal,
            )
            self.simulator.detect_contact_between(self.left_rod, sphere).using(
                SuckerActuation,
                k=0.5e1,
                nu=0.0,
                trigger=lambda arm_id=self.arm_ids[0]: self._sucker_active[arm_id],
            )
            self.simulator.detect_contact_between(self.right_rod, sphere).using(
                SuckerActuation,
                k=0.5e1,
                nu=0.0,
                trigger=lambda arm_id=self.arm_ids[1]: self._sucker_active[arm_id],
            )
            self.simulator.add_forcing_to(sphere).using(
                _PullSphereToPoint,
                target=lambda arm_id=self.arm_ids[0]: self._arm_base_position(arm_id),
                is_active=lambda arm_id=self.arm_ids[0]: self._base_pull_active[arm_id],
            )
            self.simulator.add_forcing_to(sphere).using(
                _PullSphereToPoint,
                target=lambda arm_id=self.arm_ids[1]: self._arm_base_position(arm_id),
                is_active=lambda arm_id=self.arm_ids[1]: self._base_pull_active[arm_id],
            )

        self._recording_queues = {}

        p_linear = 200.0
        p_angular = 5.0

        self.simulator.add_forcing_to(self.left_rod).using(
            TargetPoseProportionalControl,
            elem_index=-1,
            p_linear_value=p_linear,
            p_angular_value=p_angular,
            target=self.get_target_left,
            is_attached=self.is_left_attached,
            ramp_up_time=1e-3,
        )
        self.simulator.add_forcing_to(self.right_rod).using(
            TargetPoseProportionalControl,
            elem_index=-1,
            p_linear_value=p_linear,
            p_angular_value=p_angular,
            target=self.get_target_right,
            is_attached=self.is_right_attached,
            ramp_up_time=1e-3,
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

        self.simulator.detect_contact_between(self.left_rod, self.left_rod).using(
            ea.RodSelfContact, k=1e4, nu=3
        )
        self.simulator.detect_contact_between(self.right_rod, self.right_rod).using(
            ea.RodSelfContact, k=1e4, nu=3
        )
        self.simulator.detect_contact_between(self.left_rod, self.right_rod).using(
            ea.RodRodContact, k=1e4, nu=3
        )

        damping_constant = 5.0
        self.simulator.dampen(self.left_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.001,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.right_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.001,
            time_step=self.dt_internal,
        )

        self.simulator.finalize()

    def set_sucker_active(self, arm_id: str, active: bool) -> None:
        if arm_id not in self._sucker_active:
            return
        self._sucker_active[arm_id] = active

    def set_base_pull_active(self, arm_id: str, active: bool) -> None:
        if arm_id not in self._base_pull_active:
            return
        self._base_pull_active[arm_id] = active

    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        super().handle_commands(
            arm_id,
            controller_command,
            previous_controller_command=previous_controller_command,
        )
        self.set_base_pull_active(
            arm_id, bool(controller_command.buttons.get("grip_click", False))
        )
        self.set_sucker_active(
            arm_id, bool(controller_command.buttons.get("trigger_click", False))
        )

    def handle_command_inactive(self, arm_id: str) -> None:
        super().handle_command_inactive(arm_id)
        self.set_base_pull_active(arm_id, False)
        self.set_sucker_active(arm_id, False)

    def sphere_entities(self) -> list[SphereEntity]:
        spheres: list[SphereEntity] = []
        for idx, sphere in enumerate(self.spheres):
            position = np.asarray(sphere.position_collection[..., 0], dtype=np.float64)
            spheres.append(
                SphereEntity(
                    sphere_id=f"{self.user_id}_cathy_throw_sphere_{idx}",
                    owner_id=self.user_id,
                    translation=position.tolist(),
                    radius=float(sphere.radius),
                    color_rgb=[0.95, 0.62, 0.32],
                )
            )
        return spheres

    def _arm_base_position(self, arm_id: str) -> np.ndarray:
        if arm_id == self.arm_ids[0]:
            return np.asarray(
                self.left_rod.position_collection[:, 0], dtype=np.float64
            ).copy()
        if arm_id == self.arm_ids[1]:
            return np.asarray(
                self.right_rod.position_collection[:, 0], dtype=np.float64
            ).copy()
        return np.zeros(3, dtype=np.float64)

    def contact_points_for_arm(self, arm_id: str) -> list[list[float]]:
        queue = self._recording_queues.get(arm_id)
        contact_points: list[list[float]] = []
        if queue is not None:
            cutoff_time = self._time - self.contact_point_history_seconds
            while queue and queue[0][0] < cutoff_time:
                queue.popleft()
            if len(queue) > self.max_contact_points_visible:
                contact_points = [
                    point
                    for _, point in list(queue)[-self.max_contact_points_visible :]
                ]
            else:
                contact_points = [point for _, point in queue]
        return contact_points
