from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from virtual_field.core.state import MeshEntity
from virtual_field.runtime.mesh_assets import build_cylinder_gltf_data_uri
from virtual_field.runtime.mode_base import DualArmSimulationBase


@dataclass(slots=True)
class NoelObstacleSet:
    starts: np.ndarray
    directions: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray
    radii: np.ndarray


def load_noel_c4_obstacles() -> NoelObstacleSet:
    obstacle_path = (
        Path(__file__).resolve().parents[3] / "externals" / "noel-c4-obstacles.npz"
    )
    data = np.load(obstacle_path, allow_pickle=True)

    required_keys = {
        "N_OBSTACLE",
        "obstacle_direction",
        "obstacle_normal",
        "obstacle_length",
        "obstacle_radii",
        "obstacle_start",
    }
    missing = required_keys.difference(data.files)
    if missing:
        raise ValueError(f"missing Noel-C4 obstacle keys: {sorted(missing)}")

    starts = np.asarray(data["obstacle_start"], dtype=np.float64)
    directions = np.asarray(data["obstacle_direction"], dtype=np.float64)
    normals = np.asarray(data["obstacle_normal"], dtype=np.float64)
    lengths = np.asarray(data["obstacle_length"], dtype=np.float64).reshape(-1)
    radii = np.asarray(data["obstacle_radii"], dtype=np.float64).reshape(-1)
    count = int(np.asarray(data["N_OBSTACLE"]).reshape(()))

    if starts.shape != (count, 3):
        raise ValueError(f"obstacle_start must have shape ({count}, 3)")
    if directions.shape != (count, 3):
        raise ValueError(f"obstacle_direction must have shape ({count}, 3)")
    if normals.shape != (count, 3):
        raise ValueError(f"obstacle_normal must have shape ({count}, 3)")
    if lengths.shape != (count,):
        raise ValueError(f"obstacle_length must have shape ({count},)")
    if radii.shape != (count,):
        raise ValueError(f"obstacle_radii must have shape ({count},)")

    direction_norms = np.linalg.norm(directions, axis=1)
    normal_norms = np.linalg.norm(normals, axis=1)
    if np.any(direction_norms <= 0.0):
        raise ValueError("obstacle directions must be non-zero")
    if np.any(normal_norms <= 0.0):
        raise ValueError("obstacle normals must be non-zero")
    directions = directions / direction_norms[:, None]
    normals = normals / normal_norms[:, None]

    # Reframe the obstacle nest for VR: shrink uniformly, rotate the original
    # left-side placement into the viewer's forward (-Z) direction, then apply
    # an additional clockwise quarter turn in place and lift the nest center up
    # to arm height.
    scale = 0.50
    rotate_y_neg_90 = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    starts = (starts @ rotate_y_neg_90.T) * scale
    directions = directions @ rotate_y_neg_90.T
    normals = normals @ rotate_y_neg_90.T
    lengths = lengths * scale
    radii = radii * scale

    ends = starts + directions * lengths[:, None]
    center = np.vstack([starts, ends]).mean(axis=0)
    starts = (starts - center) @ rotate_y_neg_90.T + center
    directions = directions @ rotate_y_neg_90.T
    normals = normals @ rotate_y_neg_90.T

    ends = starts + directions * lengths[:, None]
    center = np.vstack([starts, ends]).mean(axis=0)
    starts[:, 0] -= 0.10
    starts[:, 1] += 1.0 - center[1] + 0.25
    starts[:, 2] -= 0.55

    return NoelObstacleSet(
        starts=starts,
        directions=directions,
        normals=normals,
        lengths=lengths,
        radii=radii,
    )


@dataclass(slots=True)
class NoelC4Simulation(DualArmSimulationBase):
    _obstacles: NoelObstacleSet = field(init=False)

    def build_simulation(self) -> None:
        import elastica as ea
        from virtual_field.runtime.spirob_elastica.constraints import (
            _SpirobBendConstraint,
        )
        from virtual_field.runtime.custom_elastica.control import (
            TargetPoseProportionalControl,
        )
        from virtual_field.runtime.spirob_elastica.sdf_objects import (
            SDFObstacleCylinders,
        )

        self._obstacles = load_noel_c4_obstacles()

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
        direction = np.array([0.0, 1.0, 0.0])
        normal = np.array([0.0, 0.0, 1.0])
        base_length = 0.551479602
        base_radius = 0.019945905
        density = 2000.0
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

        self.simulator.detect_contact_between(self.left_rod, self.right_rod).using(
            ea.RodRodContact, k=1e4, nu=3
        )
        self.simulator.detect_contact_between(self.right_rod, self.right_rod).using(
            ea.RodSelfContact, k=1e4, nu=3
        )
        self.simulator.add_forcing_to(self.left_rod).using(
            _SpirobBendConstraint,
            kt=0,
            allowed_angle_in_deg=30,
        )

        for rod in (self.left_rod, self.right_rod):
            self.simulator.add_forcing_to(rod).using(
                SDFObstacleCylinders,
                starts=self._obstacles.starts,
                directions=self._obstacles.directions,
                lengths=self._obstacles.lengths,
                radii=self._obstacles.radii,
                normals=self._obstacles.normals,
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

    def mesh_entities(self) -> list[MeshEntity]:
        meshes: list[MeshEntity] = []
        for idx in range(self._obstacles.starts.shape[0]):
            meshes.append(
                MeshEntity(
                    mesh_id=f"{self.user_id}_noel_c4_obstacle_{idx}",
                    owner_id=self.user_id,
                    asset_uri=build_cylinder_gltf_data_uri(
                        self._obstacles.starts[idx],
                        self._obstacles.directions[idx],
                        self._obstacles.normals[idx],
                        float(self._obstacles.lengths[idx]),
                        float(self._obstacles.radii[idx]),
                    ),
                )
            )
        return meshes
