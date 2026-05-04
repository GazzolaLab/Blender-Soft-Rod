from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from virtual_field.core.state import HapticEvent
from virtual_field.runtime.mode_base import DualArmSimulationBase


@dataclass(slots=True)
class SpirobsSimulation(DualArmSimulationBase):
    contact_point_history_seconds: float = 1.0
    max_contact_points_visible: int = 2000
    max_contact_points_memory: int = 8000
    tip_haptic_max_penetration: float = 0.01
    _recording_queues: dict[str, deque[tuple[float, list[float]]]] = field(
        init=False
    )
    _tip_penetration_by_arm: dict[str, float] = field(
        init=False, default_factory=dict
    )
    _haptic_events: list[HapticEvent] = field(init=False, default_factory=list)

    def build_simulation(self) -> None:
        if self.contact_point_history_seconds <= 0.0:
            raise ValueError("contact_point_history_seconds must be > 0")
        if self.max_contact_points_visible <= 0:
            raise ValueError("max_contact_points_visible must be > 0")
        if self.max_contact_points_memory <= 0:
            raise ValueError("max_contact_points_memory must be > 0")

        import elastica as ea

        from virtual_field.runtime.custom_elastica.control import (
            TargetPoseProportionalControl,
        )
        # from virtual_field.runtime.spirob_elastica.constraints import (
        #     _SpirobBendConstraint,
        # )
        from virtual_field.runtime.spirob_elastica.sdf_objects import SDFTorus

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
        base_radius = 0.02
        density = 2500.0
        youngs_modulus = 5.0e5

        self.left_rod = ea.CosseratRod.straight_rod(
            n_elem,
            np.array(self.base_left, dtype=np.float64),
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus=youngs_modulus,
        )
        self.right_rod = ea.CosseratRod.straight_rod(
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

        # contact (Comment out for now. We'll consider it later)
        self.simulator.detect_contact_between(
            self.left_rod, self.right_rod
        ).using(ea.RodRodContact, k=1e4, nu=3)
        self.simulator.detect_contact_between(
            self.left_rod, self.left_rod
        ).using(ea.RodSelfContact, k=1e4, nu=3)
        self.simulator.detect_contact_between(
            self.right_rod, self.right_rod
        ).using(ea.RodSelfContact, k=1e4, nu=3)
        # self.simulator.add_forcing_to(self.left_rod).using(
        #     _SpirobBendConstraint,
        #     kt=0,
        #     allowed_angle_in_deg=30,
        # )
        # self.simulator.add_forcing_to(self.right_rod).using(
        #     _SpirobBendConstraint,
        #     kt=2,
        # )

        torus_center = np.array([0.0, 1.0, -0.3])
        major_radius = 0.2
        minor_radius = 0.05
        recording_queue_left = deque(maxlen=self.max_contact_points_memory)
        recording_queue_right = deque(maxlen=self.max_contact_points_memory)
        self._recording_queues = {
            self.arm_ids[0]: recording_queue_left,
            self.arm_ids[1]: recording_queue_right,
        }
        self._tip_penetration_by_arm = {arm_id: 0.0 for arm_id in self.arm_ids}
        self._haptic_events = [
            HapticEvent(arm_id=arm_id, active=False, intensity=0.0)
            for arm_id in self.arm_ids
        ]
        self.simulator.add_forcing_to(self.left_rod).using(
            SDFTorus,
            center=torus_center,
            major_radius=major_radius,
            minor_radius=minor_radius,
            recording_queue=recording_queue_left,
            tip_penetration_state=self._tip_penetration_by_arm,
            tip_penetration_key=self.arm_ids[0],
        )
        self.simulator.add_forcing_to(self.right_rod).using(
            SDFTorus,
            center=torus_center,
            major_radius=major_radius,
            minor_radius=minor_radius,
            recording_queue=recording_queue_right,
            tip_penetration_state=self._tip_penetration_by_arm,
            tip_penetration_key=self.arm_ids[1],
        )

        damping_constant = 5.0
        self.simulator.dampen(self.left_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.001,
            time_step=self.dt_internal,
        )
        # self.simulator.dampen(self.left_rod).using(
        #     ea.LaplaceDissipationFilter, filter_order=5
        # )
        self.simulator.dampen(self.right_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.001,
            time_step=self.dt_internal,
        )
        # self.simulator.dampen(self.right_rod).using(
        #     ea.LaplaceDissipationFilter, filter_order=5
        # )

        self.simulator.finalize()

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
                    for _, point in list(queue)[
                        -self.max_contact_points_visible :
                    ]
                ]
            else:
                contact_points = [point for _, point in queue]
        return contact_points

    def haptic_events(self) -> list[HapticEvent]:
        for event in self._haptic_events:
            arm_id = event.arm_id
            penetration = self._tip_penetration_by_arm.get(arm_id, 0.0)
            intensity = penetration / self.tip_haptic_max_penetration
            intensity = max(0.0, min(1.0, intensity))
            event.active = intensity > 0.0
            event.intensity = intensity
        return self._haptic_events
