from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from virtual_field.core.commands import ArmCommand
from virtual_field.core.state import MeshEntity
from virtual_field.runtime.mesh_assets import build_pyvista_polydata_gltf_data_uri
from virtual_field.runtime.mode_base import DualArmSimulationBase

from importlib.resources import files

MAZE_PATH = files("virtual_field").joinpath("externals", "mesh", "pipe_maze_simple_v2.stl")


@dataclass(slots=True)
class TwoGCRSimulation(DualArmSimulationBase):
    """Dual soft arms backed by ``GrowingCR`` rods."""

    _primary_pressed: dict[str, bool] = field(init=False)
    _secondary_pressed: dict[str, bool] = field(init=False)
    _pipe_maze_asset_uri: str | None = field(init=False, default=None)

    def post_mode_setup(self) -> None:
        self._primary_pressed = {}
        self._secondary_pressed = {}
        for aid in self.arm_ids:
            self._primary_pressed[aid] = False
            self._secondary_pressed[aid] = False

    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        primary_pressed = bool(controller_command.buttons.get("primary", False))
        secondary_pressed = bool(controller_command.buttons.get("secondary", False))
        previous_primary_pressed = bool(
            previous_controller_command
            and previous_controller_command.buttons.get("primary", False)
        )
        previous_secondary_pressed = bool(
            previous_controller_command
            and previous_controller_command.buttons.get("secondary", False)
        )

        self._primary_pressed[arm_id] = primary_pressed and not previous_primary_pressed
        self._secondary_pressed[arm_id] = (
            secondary_pressed and not previous_secondary_pressed
        )

        self.set_target_pose(
            arm_id=arm_id,
            translation=controller_command.target.translation,
            rotation_xyzw=controller_command.target.rotation_xyzw,
        )

    def handle_command_inactive(self, arm_id: str) -> None:
        self._primary_pressed[arm_id] = False
        self._secondary_pressed[arm_id] = False
        super(TwoGCRSimulation, self).handle_command_inactive(arm_id)

    def build_simulation(self) -> None:
        import pyvista as pv
        import elastica as ea
        from elastica.memory_block.memory_block_rod import MemoryBlockCosseratRod

        from .custom_elastica.boundary_conditions import (
            _GrowingCRBoundaryConditions,
        )
        from .custom_elastica.rods.growing_cr import GrowingCR
        from .custom_elastica.mesh import (
            MeshSurface,
            Grid,
            RodMeshSurfaceContactGridMethod,
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
        self.simulator.enable_block_supports(GrowingCR, MemoryBlockCosseratRod)
        self.timestepper = ea.PositionVerlet()

        n_elem = 3
        total_elements = n_elem * 21
        # In viewer coordinates, forward is -Z.
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([1.0, 0.0, 0.0])
        base_length = 4.00
        base_radius = 0.015
        density = 4500.0
        youngs_modulus = 1.0e6

        left_rod_base = np.array(self.base_left, dtype=np.float64)
        self.left_rod = GrowingCR.straight_rod(
            total_elements=total_elements,
            min_elements=n_elem,
            current_elements=n_elem,
            base_position=left_rod_base,
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
            youngs_modulus=youngs_modulus,
        )
        right_rod_base = np.array(self.base_right, dtype=np.float64)
        self.right_rod = GrowingCR.straight_rod(
            total_elements=total_elements,
            min_elements=n_elem,
            current_elements=n_elem,
            base_position=right_rod_base,
            direction=direction,
            normal=normal,
            base_length=base_length,
            base_radius=base_radius,
            density=density,
            youngs_modulus=youngs_modulus,
        )
        self.simulator.append(self.left_rod)
        self.simulator.append(self.right_rod)

        # setup mesh
        pipe_mesh = pv.read(MAZE_PATH)
        pipe_mesh.scale(1.03e-2 * np.array([1.0, 1.0, 1.0]), inplace=True)
        pipe_mesh.rotate_x(-90, inplace=True)
        pipe_mesh.translate(
            -np.array(pipe_mesh.center), inplace=True
        )  # center the mesh at origin
        pipe_mesh.translate(
            np.array([0.04, 1.02, -0.45]), inplace=True
        )  # center the mesh at origin
        self._pipe_maze_asset_uri = build_pyvista_polydata_gltf_data_uri(
            pipe_mesh,
            color_rgba=(0.42, 0.48, 0.55, 0.65),
        )
        pipe_surface = MeshSurface(pipe_mesh)
        self.simulator.append(pipe_surface)

        grid = Grid(
            self.left_rod,
            pipe_surface,
            grid_dimension=3,
            exit_boundary_condition=False,
        )

        self.simulator.detect_contact_between(self.left_rod, pipe_surface).using(
            RodMeshSurfaceContactGridMethod,
            k=1e6,
            nu=1e1,
            grid=grid,
        )
        self.simulator.detect_contact_between(self.right_rod, pipe_surface).using(
            RodMeshSurfaceContactGridMethod,
            k=1e6,
            nu=1e1,
            grid=grid,
        )

        # REMOVE LATER. REPLACED BY _GrowingCRBoundaryConditions
        # self.simulator.constrain(self.left_rod).using(
        #     ea.FixedConstraint,
        #     constrained_position_idx=(0,),
        #     constrained_director_idx=(0,),
        # )
        # self.simulator.constrain(self.right_rod).using(
        #     ea.FixedConstraint,
        #     constrained_position_idx=(0,),
        #     constrained_director_idx=(0,),
        # )

        p_linear = 1000.0
        p_angular = 1.0

        # left: self.arm_ids[0]
        self.simulator.add_forcing_to(self.left_rod).using(
            _GrowingCRBoundaryConditions,
            target_position=left_rod_base,
            p_linear_value=p_linear,
            p_angular_value=p_angular,
            controller=self.get_target_left,
            trigger_increase_elements=lambda: self._secondary_pressed[self.arm_ids[0]],
            trigger_decrease_elements=lambda: self._primary_pressed[self.arm_ids[0]],
            ramp_up_time=1e-3,
        )
        # right: self.arm_ids[1]
        self.simulator.add_forcing_to(self.right_rod).using(
            _GrowingCRBoundaryConditions,
            target_position=right_rod_base,
            p_linear_value=p_linear,
            p_angular_value=p_angular,
            controller=self.get_target_right,
            trigger_increase_elements=lambda: self._secondary_pressed[self.arm_ids[1]],
            trigger_decrease_elements=lambda: self._primary_pressed[self.arm_ids[1]],
            ramp_up_time=1e-3,
        )

        # self.simulator.detect_contact_between(self.left_rod, self.right_rod).using(
        #     ea.RodRodContact, k=1e4, nu=3
        # )
        # self.simulator.detect_contact_between(self.left_rod, self.left_rod).using(
        #     ea.RodSelfContact, k=1e4, nu=3
        # )
        # self.simulator.detect_contact_between(self.right_rod, self.right_rod).using(
        #     ea.RodSelfContact, k=1e4, nu=3
        # )

        damping_constant = 5.0
        self.simulator.dampen(self.left_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.001,
            time_step=self.dt_internal,
        )
        # self.simulator.dampen(self.left_rod).using(
        #    ea.LaplaceDissipationFilter, filter_order=5
        # )
        self.simulator.dampen(self.right_rod).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=damping_constant,
            rotational_damping_constant=damping_constant * 0.001,
            time_step=self.dt_internal,
        )
        # self.simulator.dampen(self.right_rod).using(
        #    ea.LaplaceDissipationFilter, filter_order=5
        # )

        self.simulator.finalize()

    def mesh_entities(self) -> list[MeshEntity]:
        if self._pipe_maze_asset_uri is None:
            return []
        return [
            MeshEntity(
                mesh_id=f"{self.user_id}_two_gcr_pipe_maze",
                owner_id=self.user_id,
                asset_uri=self._pipe_maze_asset_uri,
            )
        ]
