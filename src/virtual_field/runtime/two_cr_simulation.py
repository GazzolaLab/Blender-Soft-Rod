from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from virtual_field.runtime.mode_base import DualArmSimulationBase


@dataclass(slots=True)
class TwoCRSimulation(DualArmSimulationBase):
    """Dual soft arms as two Cosserat rods with tip tracking and contact.

    Each arm is a straight rod with a fixed base, tip forces from
    ``TargetPoseProportionalControl``, rod–rod and self-contact, and damping.
    """

    def build_simulation(self) -> None:
        import elastica as ea

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

        n_elem = 41
        # In viewer coordinates, forward is -Z.
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([1.0, 0.0, 0.0])
        base_length = 0.55
        base_radius = 0.015
        density = 1500.0
        youngs_modulus = 8.0e5
        poisson_ratio = 0.5
        shear_modulus = youngs_modulus / (2 * (poisson_ratio + 1.0))

        self.left_rod = ea.CosseratRod.straight_rod(
            n_elem,
            np.array(self.base_left, dtype=np.float64),
            direction,
            normal,
            base_length,
            base_radius,
            density,
            youngs_modulus=youngs_modulus,
            shear_modulus=shear_modulus,
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
            shear_modulus=shear_modulus,
        )
        self.simulator.append(self.left_rod)
        self.simulator.append(self.right_rod)

        p_linear = 1000.0
        p_angular = 3.0

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

        damping_constant = 5.0
        self.simulator.dampen(self.left_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.01,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.left_rod).using(
            ea.LaplaceDissipationFilter, filter_order=5
        )
        self.simulator.dampen(self.right_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.01,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.right_rod).using(
            ea.LaplaceDissipationFilter, filter_order=5
        )

        self.simulator.finalize()
