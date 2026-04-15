from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from collections import deque

from virtual_field.runtime.mode_base import DualArmSimulationBase
from virtual_field.core.state import SphereEntity


@dataclass(slots=True)
class COOMMOctopusSimulation(DualArmSimulationBase):
    """COOMM Octopus mode"""

    def build_simulation(self) -> None:
        import elastica as ea
        from virtual_field.runtime.spirob_elastica.constraints import (
            _SpirobBendConstraint,
        )
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
        # In viewer coordinates, forward is -Z.
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([1.0, 0.0, 0.0])
        base_length = 0.55
        base_radius = 0.03
        density = 2500.0
        youngs_modulus = 2.0e6

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
        # self.simulator.detect_contact_between(self.left_rod, self.right_rod).using(
        #     ea.RodRodContact, k=1e4, nu=3
        # )
        # self.simulator.detect_contact_between(self.left_rod, self.left_rod).using(
        #     ea.RodSelfContact, k=1e4, nu=3
        # )
        # self.simulator.detect_contact_between(self.right_rod, self.right_rod).using(
        #     ea.RodSelfContact, k=1e4, nu=3
        # )

        self.simulator.add_forcing_to(self.left_rod).using(
            ea.GravityForces,
            acc_gravity=np.array([0.0, -9.81, 0.0]),
        )
        self.simulator.add_forcing_to(self.right_rod).using(
            ea.GravityForces,
            acc_gravity=np.array([0.0, -1.00, 0.0]),
        )

        damping_constant = 5.0
        self.simulator.dampen(self.left_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.1,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.left_rod).using(
            ea.LaplaceDissipationFilter, filter_order=5
        )
        self.simulator.dampen(self.right_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.1,
            time_step=self.dt_internal,
        )
        self.simulator.dampen(self.right_rod).using(
            ea.LaplaceDissipationFilter, filter_order=5
        )

        self.simulator.finalize()

    def sphere_entities(self) -> list[SphereEntity]:
        spheres: list[SphereEntity] = []
        for idx, sphere in enumerate(self.spheres):
            position = np.asarray(sphere.position_collection[..., 0], dtype=np.float64)
            spheres.append(
                SphereEntity(
                    sphere_id=f"{self.user_id}_coomm_octopus_sphere_{idx}",
                    owner_id=self.user_id,
                    translation=position.tolist(),
                    radius=float(sphere.radius),
                    color_rgb=[0.95, 0.62, 0.32],
                )
            )
        return spheres
