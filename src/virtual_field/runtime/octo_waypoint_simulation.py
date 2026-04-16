from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger
import numpy as np
import elastica as ea

from virtual_field.core.commands import ArmCommand, MultiArmCommand
from virtual_field.core.state import SphereEntity, Transform, MeshEntity
from virtual_field.runtime.custom_elastica.dissipation import RayleighDamping
from virtual_field.runtime.foraging_elastica import (
    SphereHeadTether,
    BaseSphereTether,
    OctoArmPolicy,
    SegmentExtensionActuation,
    SuckerActuation,
    YSurfaceBallwGravity,
    current_activation,
    idle_policy_like,
    rotate_policy_by_angle,
)
from virtual_field.runtime.mode_base import OctoArmSimulationBase
from virtual_field.runtime.orientation import controller_quat_xyzw_to_matrix
from virtual_field.runtime.spirob_elastica.spirob import create_spirob
from virtual_field.runtime.mesh_assets import build_pyvista_polydata_gltf_data_uri
from importlib.resources import files

# Configurations
ASSET_PATH = files("virtual_field").joinpath("externals", "crawling")
EXTERNAL_POLICY_PATH = ASSET_PATH / "best_policy.npy"
EXTERNAL_MESH_PATH = ASSET_PATH / "terrain" / "scene.gltf"
EXTERNAL_MESH_BASE_COLOR_TEXTURE_PATH = (
    ASSET_PATH / "terrain" / "textures" / "m32_Viekoda_Bay_baseColor.jpeg"
)

WAYPOINT_PLANE_Y = -0.05
WAYPOINT_RADIUS = 0.04
WAYPOINT_COLOR_RGB = [0.98, 0.74, 0.24]
SPHERE_COLOR_RGB = [0.55, 0.45, 0.95]
# Must match VR/client/app.js `initialControllerForward` (Three.js local axis used for ray).
RIGHT_CONTROLLER_FORWARD = np.array([0.0, -1.0, 0.0], dtype=np.float64)
# Must match VR/client/app.js `waypointPlaneConfig` (centerX/Y/Z, sizeX/Z).
WAYPOINT_PLANE_CENTER = np.array([0.0, WAYPOINT_PLANE_Y, -0.3], dtype=np.float64)
WAYPOINT_PLANE_SIZE_X = 4.0
WAYPOINT_PLANE_SIZE_Z = 3.0
WAYPOINT_REACHED_RADIUS = 0.08
MAX_WAYPOINT_QUEUE = 8
WAYPOINT_QUEUE_COLOR_RGB = [0.98, 0.74, 0.24]
WAYPOINT_CURRENT_COLOR_RGB = [0.35, 0.95, 0.45]
# Crawling policy has eight tentacle channels; ``arm_8`` is the head rod (kinematic).
TENTACLE_COUNT = 8


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


def _pentagon_waypoints_xy_plane(radius: float) -> np.ndarray:
    """Five targets on the y=0 plane, matching ``research/.../fast_octo_arm``."""

    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=True)[:-1]
    return np.asarray(
        [[radius * np.sin(angle), 0.0, -radius * np.cos(angle)] for angle in angles],
        dtype=np.float64,
    )


@dataclass(slots=True)
class OctoWaypointSimulation(OctoArmSimulationBase):
    """Waypoint crawl mode.

    Parameters
    ----------
    seed_pentagon_waypoints
        If True (default), prefill the queue with five pentagon vertices like
        ``fast_octo_arm._pentagon_waypoints`` so the body visits them in order.
    preset_pentagon_radius
        Horizontal radius in meters for that pentagon (``fast_octo`` default 0.45).
    enable_controller_trigger_waypoints
        If True, controller trigger adds projected waypoints. Disabled by default
        while debugging preset navigation.
    """

    seed_pentagon_waypoints: bool = True
    preset_pentagon_radius: float = 0.45
    enable_controller_trigger_waypoints: bool = False

    head: ea.CosseratRod = field(init=False)
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
    _waypoint_queue: list[np.ndarray] = field(init=False, default_factory=list)
    _waypoint_active: bool = field(init=False, default=False)
    _cycle_heading_angle: float = field(init=False, default=0.0)
    _last_cycle_index: int = field(init=False, default=-1)
    _crawl_active_until: float = field(init=False, default=0.0)
    _loco_coast_until: float | None = field(init=False, default=None)
    _defer_heading_until: float | None = field(init=False, default=None)
    _last_trigger_pressed: dict[str, bool] = field(init=False, default_factory=dict)
    base_suction_active: np.ndarray = field(init=False)
    middle_suction_active: np.ndarray = field(init=False)
    target_extension: np.ndarray = field(init=False)
    target_stiffness: np.ndarray = field(init=False)
    target_bend: np.ndarray = field(init=False)
    _terrain_asset_uri: str | None = field(init=False, default=None)

    def build_simulation(self) -> None:
        import pyvista as pv
        from .custom_elastica.mesh import (
            MeshSurface,
            Grid,
            RodMeshSurfaceContactGridMethodWithAnisotropicFriction,
        )

        self.simulator = _Simulator()
        self.timestepper = ea.PositionVerlet()

        self.head = _make_head(np.array([0.0, 0.0, 0.0], dtype=np.float64))
        self.simulator.append(self.head)

        self._head_local_positions = np.asarray(
            self.head.position_collection, dtype=np.float64
        ).copy() - np.asarray(self.head.position_collection[:, [0]], dtype=np.float64)
        self._head_pose = Transform(
            translation=list(np.array([0.0, 0.0, 0.0], dtype=np.float64)),
            rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        )

        self.base_policy = _load_best_policy()
        self.idle_policy = idle_policy_like(self.base_policy)
        self.active_policy = self.idle_policy
        self._last_trigger_pressed = {"left": False, "right": False}

        self.base_suction_active = np.zeros(TENTACLE_COUNT, dtype=np.float64)
        self.middle_suction_active = np.zeros(TENTACLE_COUNT, dtype=np.float64)
        self.target_extension = np.zeros(TENTACLE_COUNT, dtype=np.float64)
        self.target_stiffness = np.ones(TENTACLE_COUNT, dtype=np.float64)
        self.target_bend = np.zeros(TENTACLE_COUNT, dtype=np.float64)

        self.base_sphere = ea.Sphere(
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            self.base_sphere_radius,
            100.0,
        )
        self.simulator.append(self.base_sphere)
        self.simulator.detect_contact_between(self.base_sphere, self.head).using(
            SphereHeadTether,
            head_orientation=self.head.director_collection[..., 0],
        )
        self.simulator.dampen(self.head).using(
            ea.AnalyticalLinearDamper,
            translational_damping_constant=2.0,
            rotational_damping_constant=3.0e-3,
            time_step=self.dt_internal,
        )
        self.simulator.add_forcing_to(self.base_sphere).using(
            YSurfaceBallwGravity,
            k_c=1.0e5,
            nu_c=1.0e1,
            plane_origin=-self.base_sphere_radius,
        )
        self.simulator.dampen(self.base_sphere).using(
            RayleighDamping,
            damping_constant=1.0,
            rotational_damping_constant=1.0e-2,
            time_step=self.dt_internal,
        )

        plane_y = -0.02
        terrain_mesh = pv.read(EXTERNAL_MESH_PATH)
        terrain_mesh = terrain_mesh["Node_0"].extract_surface(algorithm=None)
        terrain_mesh.translate(
            -np.array(terrain_mesh.center), inplace=True
        )  # center the mesh at origin
        terrain_mesh.scale(1.0 * np.array([1, 1, 1]), inplace=True)  # rescale mesh
        # terrain_mesh.rotate_x(
        #     90, inplace=True
        # )  # rotate so surface upper side points in +y
        terrain_mesh.translate(
            np.array([0, plane_y - terrain_mesh.bounds[3], 0])
        )  # surface top point at plane_y
        ground_surface = MeshSurface(terrain_mesh)
        self.simulator.append(ground_surface)
        self._terrain_asset_uri = build_pyvista_polydata_gltf_data_uri(
            terrain_mesh,
            color_rgba=(0.42, 0.48, 0.55, 1.00),
            base_color_texture_path=EXTERNAL_MESH_BASE_COLOR_TEXTURE_PATH,
        )
        dummy_rod = create_spirob(
            15,
            np.array([0.0, 0.0, 0.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            0.45,
            0.02,
            5500.0,
            5.0e5,
        )
        grid = Grid(
            rod=dummy_rod,
            surface=ground_surface,
            grid_dimension=2,
            exit_boundary_condition=False,
            grid_axes=[0, 2],
        )

        rods: dict[str, ea.CosseratRod] = {}
        rods[self.arm_ids[TENTACLE_COUNT]] = self.head
        for arm_index in range(TENTACLE_COUNT):
            arm_id = self.arm_ids[arm_index]
            angle = 2.0 * np.pi * (arm_index + 0.5) / TENTACLE_COUNT
            direction = np.array(
                [np.sin(angle), 0.0, -np.cos(angle)],
                dtype=np.float64,
            )
            rod = create_spirob(
                15,
                np.array([0.0, 0.0, 0.0], dtype=np.float64),
                direction,
                np.array([0.0, -1.0, 0.0], dtype=np.float64),
                0.45,
                0.02,
                5500.0,
                5.0e5,
            )
            rods[arm_id] = rod
            self.simulator.append(rod)
            self.simulator.detect_contact_between(rod, self.base_sphere).using(
                BaseSphereTether,
                k=5.0e3,
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
            self.simulator.dampen(rod).using(
                ea.AnalyticalLinearDamper,
                translational_damping_constant=2.0,
                rotational_damping_constant=3.0e-3,
                time_step=self.dt_internal,
            )
            self.simulator.detect_contact_between(rod, ground_surface).using(
                RodMeshSurfaceContactGridMethodWithAnisotropicFriction,
                k=5e2,
                nu=1e-1,
                gamma=0.1,
                grid=grid,
                slip_velocity_tol=1e-4,
                static_mu_array=np.array([0, 0, 0]),
                kinetic_mu_array=np.array([0.1, 0.1, 0.1]),
            )

        self.rods = rods
        self.simulator.finalize()

        # DEBUG
        # if self.seed_pentagon_waypoints:
        #     pts = _pentagon_waypoints_xy_plane(float(self.preset_pentagon_radius))
        #     self._waypoint_queue = [pts[i].copy() for i in range(min(5, len(pts)))]
        #     logger.info(
        #         "octo-waypoint user={} seeded {} preset pentagon waypoints (r={})",
        #         self.user_id,
        #         len(self._waypoint_queue),
        #         self.preset_pentagon_radius,
        #     )

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
        self._update_waypoint_from_command(command)

    def _update_waypoint_from_command(self, command: MultiArmCommand) -> None:
        if not self.enable_controller_trigger_waypoints:
            return

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
                waypoint = self._project_waypoint_from_transform(
                    controller_command.target
                )
                if (
                    waypoint is not None
                    and len(self._waypoint_queue) < MAX_WAYPOINT_QUEUE
                ):
                    self._waypoint_queue.append(waypoint)
                    logger.info(
                        "octo-waypoint user={} {} trigger: enqueued waypoint "
                        "[{:.4f}, {:.4f}, {:.4f}] (queue_len={})",
                        self.user_id,
                        hand,
                        float(waypoint[0]),
                        float(waypoint[1]),
                        float(waypoint[2]),
                        len(self._waypoint_queue),
                    )
                elif waypoint is None:
                    logger.debug(
                        "octo-waypoint user={} {} trigger: ray did not yield a waypoint",
                        self.user_id,
                        hand,
                    )
                else:
                    logger.debug(
                        "octo-waypoint user={} {} trigger: queue full (max={})",
                        self.user_id,
                        hand,
                        MAX_WAYPOINT_QUEUE,
                    )
            self._last_trigger_pressed[hand] = pressed

    def _is_waypoint_reached(self, waypoint: np.ndarray) -> bool:
        sphere_position = self.base_sphere.position_collection[:, 0]
        horizontal_delta = np.asarray(
            [
                waypoint[0] - sphere_position[0],
                waypoint[2] - sphere_position[2],
            ],
            dtype=np.float64,
        )
        return float(np.linalg.norm(horizontal_delta)) <= WAYPOINT_REACHED_RADIUS

    def _project_waypoint_from_transform(
        self, transform: Transform
    ) -> np.ndarray | None:
        origin = np.asarray(transform.translation, dtype=np.float64)
        rotation = controller_quat_xyzw_to_matrix(transform.rotation_xyzw).T
        direction = rotation @ RIGHT_CONTROLLER_FORWARD
        direction_y = float(direction[1])
        if (not np.isfinite(direction_y)) or direction_y >= -1.0e-6:
            return None

        t_hit = (WAYPOINT_PLANE_Y - float(origin[1])) / direction_y
        if (not np.isfinite(t_hit)) or t_hit < 0.0:
            return None

        waypoint = origin + t_hit * direction
        waypoint[1] = WAYPOINT_PLANE_Y
        if not np.all(np.isfinite(waypoint)):
            return None
        half_x = 0.5 * WAYPOINT_PLANE_SIZE_X
        half_z = 0.5 * WAYPOINT_PLANE_SIZE_Z
        if (
            waypoint[0] < WAYPOINT_PLANE_CENTER[0] - half_x
            or waypoint[0] > WAYPOINT_PLANE_CENTER[0] + half_x
            or waypoint[2] < WAYPOINT_PLANE_CENTER[2] - half_z
            or waypoint[2] > WAYPOINT_PLANE_CENTER[2] + half_z
        ):
            return None
        return waypoint

    def _heading_angle_from_waypoint(self, waypoint: np.ndarray) -> float:
        sphere_position = np.asarray(
            self.base_sphere.position_collection[:, 0],
            dtype=np.float64,
        )
        displacement = np.asarray(
            [waypoint[0] - sphere_position[0], waypoint[2] - sphere_position[2]],
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(displacement))
        if not np.isfinite(norm) or norm <= 1.0e-9:
            return self._cycle_heading_angle
        displacement /= norm
        return float(np.arctan2(displacement[0], -displacement[1]))

    def _queue_head_is_reached(self) -> bool:
        """True if the queue is empty or the front waypoint is inside reach radius."""
        if not self._waypoint_queue:
            return True
        return self._is_waypoint_reached(self._waypoint_queue[0])

    def _drain_reached_waypoints(self) -> list[str]:
        """Remove consecutive reached waypoints from the front; return labels for logging."""
        labels: list[str] = []
        while self._waypoint_queue and self._queue_head_is_reached():
            w = self._waypoint_queue.pop(0)
            labels.append(f"[{float(w[0]):.4f},{float(w[1]):.4f},{float(w[2]):.4f}]")
        return labels

    def _log_waypoint_pops(self, popped_labels: list[str]) -> None:
        if not popped_labels:
            return
        logger.info(
            "octo-waypoint user={} reached {} waypoint(s) {}; queue_len={}",
            self.user_id,
            len(popped_labels),
            popped_labels,
            len(self._waypoint_queue),
        )

    def _sync_coast_deadline(
        self, was_chasing: bool, has_target: bool, period: float
    ) -> None:
        """Update ``_waypoint_active`` and coast-to-end-of-period deadline when idle."""
        self._waypoint_active = has_target
        if has_target:
            self._loco_coast_until = None
            return
        self._defer_heading_until = None
        if was_chasing:
            self._loco_coast_until = float(
                (np.floor(self._time / period) + 1.0) * period
            )
        elif (
            self._loco_coast_until is not None and self._time >= self._loco_coast_until
        ):
            self._loco_coast_until = None

    def _locomotion_is_active(self) -> bool:
        return self._waypoint_active or (
            self._loco_coast_until is not None and self._time < self._loco_coast_until
        )

    def _set_active_policy_toward_waypoint(
        self, waypoint: np.ndarray, cycle_index: int
    ) -> None:
        self._cycle_heading_angle = self._heading_angle_from_waypoint(waypoint)
        self.active_policy = rotate_policy_by_angle(
            self.base_policy, self._cycle_heading_angle
        )
        self._last_cycle_index = cycle_index

    def _refresh_heading_and_active_policy(
        self, waypoint: np.ndarray, period: float
    ) -> None:
        """Align ``active_policy`` with the queue head when the schedule allows.

        Heading is fixed within one ``T_L`` interval. After a waypoint is *touched*
        (popped) but a next target remains, the **current** policy is held until the
        **end of that** ``T_L`` period; only then we rotate toward the new head.
        """
        cycle_index = int(np.floor(self._time / period))
        self._crawl_active_until = (cycle_index + 1) * period

        if self._defer_heading_until is not None:
            if self._time < self._defer_heading_until:
                return
            self._defer_heading_until = None
            self._set_active_policy_toward_waypoint(waypoint, cycle_index)
            return

        if (
            cycle_index != self._last_cycle_index
            or self.active_policy is self.idle_policy
        ):
            self._set_active_policy_toward_waypoint(waypoint, cycle_index)

    def _write_tentacle_actuation(self, phase: float, policy: OctoArmPolicy) -> None:
        for arm_index in range(TENTACLE_COUNT):
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
        was_chasing = self._waypoint_active
        popped = self._drain_reached_waypoints()
        self._log_waypoint_pops(popped)

        period = float(self.base_policy.T_L)
        has_target = bool(self._waypoint_queue)
        self._sync_coast_deadline(was_chasing, has_target, period)

        if popped and has_target:
            self._defer_heading_until = float(
                (np.floor(self._time / period) + 1.0) * period
            )

        if self._locomotion_is_active():
            phase = float((self._time % period) / period)
            if has_target:
                self._refresh_heading_and_active_policy(self._waypoint_queue[0], period)
            policy = self.active_policy
        else:
            phase = 0.0
            policy = self.idle_policy
            self._last_cycle_index = -1

        self._write_tentacle_actuation(phase, policy)

    def step(self, dt: float) -> None:
        """
        (Note) Assume dt is always positive and not too small (probably > 1e-4).
        """
        substeps = max(1, int(np.ceil(dt / self.dt_internal)))
        step_dt = dt / substeps  # Actual dt
        for _ in range(substeps):
            self._apply_policy()
            self._time = self.timestepper.step(self.simulator, self._time, step_dt)

    def sphere_entities(self) -> list[SphereEntity]:
        spheres: list[SphereEntity] = []

        # Add center sphere
        # spheres.append(
        #     SphereEntity(
        #         sphere_id=f"{self.user_id}_waypoint_center",
        #         owner_id=self.user_id,
        #         translation=self.base_sphere.position_collection[:, 0].tolist(),
        #         radius=self.base_sphere_radius,
        #         color_rgb=SPHERE_COLOR_RGB,
        #         visible=True,
        #     )
        # )

        for idx in range(MAX_WAYPOINT_QUEUE):
            waypoint = (
                self._waypoint_queue[idx] if idx < len(self._waypoint_queue) else None
            )
            spheres.append(
                SphereEntity(
                    sphere_id=f"{self.user_id}_waypoint_{idx}",
                    owner_id=self.user_id,
                    translation=(
                        waypoint.tolist()
                        if waypoint is not None
                        else WAYPOINT_PLANE_CENTER.tolist()
                    ),
                    radius=WAYPOINT_RADIUS,
                    color_rgb=(
                        WAYPOINT_CURRENT_COLOR_RGB
                        if idx == 0 and waypoint is not None
                        else WAYPOINT_QUEUE_COLOR_RGB
                    ),
                    visible=waypoint is not None,
                )
            )

        return spheres

    def mesh_entities(self) -> list[MeshEntity]:
        if self._terrain_asset_uri is None:
            return []
        return [
            MeshEntity(
                mesh_id=f"{self.user_id}_waypoint_terrain",
                owner_id=self.user_id,
                asset_uri=self._terrain_asset_uri,
            )
        ]
