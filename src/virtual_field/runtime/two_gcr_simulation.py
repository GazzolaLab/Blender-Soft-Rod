from __future__ import annotations

from typing import Any

from dataclasses import dataclass, field
from importlib.resources import files

import numpy as np

from virtual_field.core.commands import ArmCommand
from virtual_field.core.state import MeshEntity, SphereEntity
from virtual_field.runtime.mesh_assets import (
    build_pyvista_polydata_gltf_data_uri,
)
from virtual_field.runtime.mode_base import DualArmSimulationBase

from .custom_elastica.contacts import TipSuctionToSphere
from .custom_elastica.dissipation import RayleighDamping

MAZE_PATH = files("virtual_field").joinpath(
    "externals", "mesh", "pipe_maze_simple_with_marker.stl"
)
# MAZE_PATH = files("virtual_field").joinpath("externals", "mesh", "pipe_maze_complex_with_marker.stl")


@dataclass(slots=True)
class TwoGCRSimulation(DualArmSimulationBase):
    """Dual soft arms backed by ``GrowingCR`` rods."""

    spheres: list[Any] = field(init=False)
    _sucker_active: dict[str, bool] = field(init=False, default_factory=dict)
    _primary_pressed: dict[str, bool] = field(init=False, default_factory=dict)
    _secondary_pressed: dict[str, bool] = field(init=False, default_factory=dict)
    _grip_wave_event: dict[str, int] = field(init=False, default_factory=dict)
    _joystick_command: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    _pipe_maze_asset_uri: str | None = field(init=False, default=None)

    def _initialize_control_state(self) -> None:
        self._primary_pressed = {}
        self._secondary_pressed = {}
        self._grip_wave_event = {}
        self._joystick_command = {}
        for aid in self.arm_ids:
            self._primary_pressed[aid] = False
            self._secondary_pressed[aid] = False
            self._grip_wave_event[aid] = 0
            self._joystick_command[aid] = np.zeros(2, dtype=np.float64)

    def post_mode_setup(self) -> None:
        self._initialize_control_state()

    def handle_commands(
        self,
        arm_id: str,
        controller_command: ArmCommand,
        previous_controller_command: ArmCommand | None = None,
    ) -> None:
        primary_pressed = bool(controller_command.buttons.get("primary", False))
        secondary_pressed = bool(controller_command.buttons.get("secondary", False))
        grip_pressed = bool(controller_command.buttons.get("grip_click", False))
        previous_grip_pressed = bool(
            previous_controller_command
            and previous_controller_command.buttons.get("grip_click", False)
        )

        # Mirror hardware: pressed → true, released → false. Both held → no grow/retract.
        if primary_pressed and secondary_pressed:
            self._primary_pressed[arm_id] = False
            self._secondary_pressed[arm_id] = False
        else:
            self._primary_pressed[arm_id] = primary_pressed
            self._secondary_pressed[arm_id] = secondary_pressed
        if grip_pressed and not previous_grip_pressed:
            self._grip_wave_event[arm_id] += 1
        self._joystick_command[arm_id][0] = controller_command.joystick[0]
        self._joystick_command[arm_id][1] = controller_command.joystick[1]

        self.set_target_pose(
            arm_id=arm_id,
            translation=controller_command.target.translation,
            rotation_xyzw=controller_command.target.rotation_xyzw,
        )

        self._set_sucker_active(
            arm_id, bool(controller_command.buttons.get("trigger_click", False))
        )

    def handle_command_inactive(self, arm_id: str) -> None:
        self._primary_pressed[arm_id] = False
        self._secondary_pressed[arm_id] = False
        self._joystick_command[arm_id].fill(0.0)
        self._set_sucker_active(arm_id, False)

    def build_simulation(self) -> None:
        import elastica as ea
        import pyvista as pv
        from elastica.memory_block.memory_block_rod import (
            MemoryBlockCosseratRod,
        )

        from .custom_elastica.boundary_conditions import (
            _GrowingCRBoundaryConditions,
        )
        from .custom_elastica.forcing import (
            _TipJoystickBending,
            _TravelingContractingWave,
        )
        from .custom_elastica.mesh import (
            Grid,
            MeshSurface,
            RodMeshSurfaceContactGridMethod,
            SphereMeshSurfaceContact,
        )
        from .custom_elastica.rods.growing_cr import GrowingCR

        class _Simulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
            ea.Contact,
        ):
            pass

        self._initialize_control_state()
        self.simulator = _Simulator()
        self.simulator.enable_block_supports(GrowingCR, MemoryBlockCosseratRod)
        self.timestepper = ea.PositionVerlet()

        initial_active_elements = 6
        total_elements = initial_active_elements * 21
        # In viewer coordinates, forward is -Z.
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([0.0, -1.0, 0.0])
        base_length = 4.00
        base_radius = 0.015
        density = 4500.0
        youngs_modulus = 1.0e6

        left_rod_base = np.array(self.base_left, dtype=np.float64)
        self.left_rod = GrowingCR.straight_rod(
            total_elements=total_elements,
            min_elements=initial_active_elements,
            current_elements=initial_active_elements,
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
            min_elements=initial_active_elements,
            current_elements=initial_active_elements,
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
        maze_mesh = pv.read(MAZE_PATH)

        maze_mesh.scale(1.03e-2 * np.array([1.0, 1.0, 1.0]), inplace=True)
        maze_mesh.rotate_x(-90, inplace=True)

        maze_mesh.translate(
            -np.array(maze_mesh.center), inplace=True
        )  # center the mesh at origin

        # move so that max mesh value is a z=0
        maze_mesh.translate(np.array([0, 0, maze_mesh.bounds[4]]), inplace=True)

        # find vertex indices for the enterance holes
        hole_vertex_idx = maze_mesh.points[:, 2] > -0.01 * (
            maze_mesh.bounds[5] - maze_mesh.bounds[4]
        )  # mesh vertices between z=0 and z=-0.01*total_z_extent
        hole_vertices_center_x = 0.5 * (
            max(maze_mesh.points[hole_vertex_idx, 0])
            + min(maze_mesh.points[hole_vertex_idx, 0])
        )
        hole_vertices_center_y = 0.5 * (
            max(maze_mesh.points[hole_vertex_idx, 1])
            + min(maze_mesh.points[hole_vertex_idx, 1])
        )

        # move mesh so that origin is between hole centers
        maze_mesh.translate(
            np.array([-hole_vertices_center_x, -hole_vertices_center_y, 0]),
            inplace=True,
        )

        # finally move origin to center of two bases so arms are inside hole centers
        maze_mesh.translate(0.50 * (right_rod_base + left_rod_base), inplace=True)

        # Label connected components
        conn = maze_mesh.connectivity()

        # Extract number of regions
        region_ids = conn["RegionId"]
        n_regions = region_ids.max() + 1

        components = []
        component_volumes = []
        for i in range(n_regions):
            comp = conn.threshold([i, i], scalars="RegionId").extract_surface(
                algorithm="dataset_surface"
            )
            component_volumes.append(comp.volume)
            components.append(comp)

        target_reduction = 0.9  # reduce number of faces by this ratio
        pipe_mesh = components.pop(np.argmax(component_volumes)).decimate(
            target_reduction
        )  # largest volume will be the pipe
        self._pipe_maze_asset_uri = build_pyvista_polydata_gltf_data_uri(
            pipe_mesh,
            color_rgba=(0.42, 0.48, 0.55, 0.55),
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

        self._sucker_active = {
            self.arm_ids[0]: False,
            self.arm_ids[1]: False,
        }

        self.spheres = []
        sphere_radius = 0.025
        sphere_density = 200.0
        for comp in components:
            sphere = ea.Sphere(np.array(comp.center), sphere_radius, sphere_density)
            self.spheres.append(sphere)
            self.simulator.append(sphere)
            self.simulator.detect_contact_between(self.left_rod, sphere).using(
                ea.RodSphereContact, k=2e3, nu=2.0
            )
            self.simulator.detect_contact_between(self.right_rod, sphere).using(
                ea.RodSphereContact, k=2e3, nu=2.0
            )
            damping_constant = 1e0
            self.simulator.dampen(sphere).using(
                RayleighDamping,
                damping_constant=damping_constant,
                rotational_damping_constant=damping_constant * 1.0,
                time_step=self.dt_internal,
            )
            self.simulator.add_forcing_to(sphere).using(
                ea.GravityForces,
                acc_gravity=np.array([0.0, -9.80665, 0.0]),
            )
            self.simulator.detect_contact_between(sphere, pipe_surface).using(
                SphereMeshSurfaceContact,
                k=5e3,
                nu=5e0,
                search_radius=5 * sphere_radius,
            )

            # tip_index = np.array([self.left_rod.n_elems - 1])
            self.simulator.detect_contact_between(self.left_rod, sphere).using(
                TipSuctionToSphere,
                # SuckerActuationToSphere,
                k=1.0e1,
                nu=1.0,
                trigger=lambda arm_id=self.arm_ids[0]: self._sucker_active[arm_id],
                # sucker_index=tip_index,
            )
            self.simulator.detect_contact_between(self.right_rod, sphere).using(
                TipSuctionToSphere,
                # SuckerActuationToSphere,
                k=1.0e1,
                nu=1.0,
                trigger=lambda arm_id=self.arm_ids[1]: self._sucker_active[arm_id],
                # sucker_index=tip_index,
            )

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
        self.simulator.add_forcing_to(self.left_rod).using(
            _TipJoystickBending,
            joystick=lambda arm_id=self.arm_ids[0]: self._joystick_command[arm_id],
            gain=0.15,
            smoothing=0.18,
            controlled_elements=8,
        )
        self.simulator.add_forcing_to(self.left_rod).using(
            _TravelingContractingWave,
            event_id=lambda arm_id=self.arm_ids[0]: self._grip_wave_event.get(
                arm_id, 0
            ),
            original_rest_sigma=self.left_rod.rest_sigma,
            original_shear_matrix=self.left_rod.shear_matrix,
            original_bend_matrix=self.left_rod.bend_matrix,
            amplitude=-0.15,
            stiffness_amplitude=0.40,
            width=3.5,
            duration=0.50,
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
        self.simulator.add_forcing_to(self.right_rod).using(
            _TipJoystickBending,
            joystick=lambda arm_id=self.arm_ids[1]: self._joystick_command[arm_id],
            gain=0.15,
            smoothing=0.18,
            controlled_elements=8,
        )
        self.simulator.add_forcing_to(self.right_rod).using(
            _TravelingContractingWave,
            event_id=lambda arm_id=self.arm_ids[1]: self._grip_wave_event.get(
                arm_id, 0
            ),
            original_rest_sigma=self.right_rod.rest_sigma,
            original_shear_matrix=self.right_rod.shear_matrix,
            original_bend_matrix=self.right_rod.bend_matrix,
            amplitude=-0.15,
            stiffness_amplitude=0.40,
            width=3.5,
            duration=0.50,
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
                static_asset=True,
            )
        ]

    def sphere_entities(self) -> list[SphereEntity]:
        spheres: list[SphereEntity] = []
        for idx, sphere in enumerate(self.spheres):
            position = np.asarray(sphere.position_collection[..., 0], dtype=np.float64)
            spheres.append(
                SphereEntity(
                    sphere_id=f"{self.user_id}_pipe_sphere_{idx}",
                    owner_id=self.user_id,
                    translation=position.tolist(),
                    radius=float(sphere.radius),
                    color_rgb=[0.95, 0.62, 0.32],
                )
            )
        return spheres

    def _set_sucker_active(self, arm_id: str, active: bool) -> None:
        if arm_id not in self._sucker_active:
            return
        self._sucker_active[arm_id] = active
