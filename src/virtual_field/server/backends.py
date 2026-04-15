from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin

from loguru import logger

from virtual_field.core.commands import ArmCommand, MultiArmCommand
from virtual_field.core.state import (
    ArmState,
    MeshEntity,
    OverlayPointsEntity,
    SceneState,
    SphereEntity,
    Transform,
)
from virtual_field.runtime.mode_base import (
    DualArmSimulationBase,
    OctoArmSimulationBase,
    SimulationBase,
)
from virtual_field.runtime.mode_registry import get_mode_spec


def _default_arm_state(arm_id: str, owner_user_id: str, base: Transform) -> ArmState:
    tip = Transform(
        translation=[
            base.translation[0],
            base.translation[1],
            base.translation[2] + 0.4,
        ],
        rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    centerline = [
        [base.translation[0], base.translation[1], base.translation[2]],
        [base.translation[0], base.translation[1], base.translation[2] + 0.2],
        [base.translation[0], base.translation[1], base.translation[2] + 0.4],
    ]
    element_lengths = [0.2, 0.2]
    return ArmState(
        arm_id=arm_id,
        owner_user_id=owner_user_id,
        base=base,
        tip=tip,
        centerline=centerline,
        radii=[0.04, 0.03],
        element_lengths=element_lengths,
        directors=[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
    )


def _allocate_linear_bases(
    user_id: str,
    *,
    arm_count: int,
    arm_spacing: float,
    base_x: float,
    base_y: float,
    base_z: float,
) -> dict[str, Transform]:
    origin = base_x - 0.5 * arm_spacing * (arm_count - 1)
    return {
        f"{user_id}_arm_{index}": Transform(
            translation=[origin + index * arm_spacing, base_y, base_z],
            rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        for index in range(arm_count)
    }


def _allocate_octo_bases(
    user_id: str,
    *,
    base_x: float,
    base_y: float,
    base_z: float,
) -> dict[str, Transform]:
    """Eight tentacles on a ring plus a ninth ``arm_8`` at the body center (head)."""
    ring_radius = 0.32
    bases: dict[str, Transform] = {
        f"{user_id}_arm_{index}": Transform(
            translation=[
                base_x + ring_radius * cos(-0.5 * pi + 2.0 * pi * index / 8.0),
                base_y,
                base_z + ring_radius * sin(-0.5 * pi + 2.0 * pi * index / 8.0),
            ],
            rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        for index in range(8)
    }
    bases[f"{user_id}_arm_8"] = Transform(
        translation=[base_x, base_y, base_z],
        rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
    )
    return bases


@dataclass(slots=True)
class MultiArmPassThroughBackend:
    """Server-side scene backend for multiple users and arms.

    Holds per-arm ``ArmState``, meshes, overlay points, and spheres. Each
    :meth:`step` applies controller commands and returns a ``SceneState``
    snapshot.

    If the user's ``character_mode`` is listed in ``SIMULATION_FACTORIES``, a
    ``DualArmSimulation`` is created: physics stepping, targets, and attachments
    are delegated to it, and its meshes/spheres are merged here.

    Otherwise arm poses are updated in pass-through fashion: the tip and a
    simple centerline follow the controller target without elastica.
    """

    _timestamp: float = field(init=False, default=0.0)
    _arms: dict[str, ArmState] = field(init=False, default_factory=dict)
    _user_arms: dict[str, list[str]] = field(init=False, default_factory=dict)
    _user_mode: dict[str, str] = field(init=False, default_factory=dict)
    _meshes: dict[str, MeshEntity] = field(init=False, default_factory=dict)
    _overlay_points: dict[str, OverlayPointsEntity] = field(
        init=False, default_factory=dict
    )
    _spheres: dict[str, SphereEntity] = field(init=False, default_factory=dict)
    _simulations: dict[str, SimulationBase] = field(
        init=False, default_factory=dict
    )
    _previous_commands: dict[str, ArmCommand] = field(init=False, default_factory=dict)

    def register_user(
        self,
        user_id: str,
        character_mode: str = "demo-spline",
        requested_arm_count: int | None = None,
        arm_spacing: float = 0.3,
        base_x: float = 0.0,
        base_y: float = 1.0,
        base_z: float = -0.15,
    ) -> list[str]:
        """Register a user and allocate arm ids.

        Parameters
        ----------
        user_id : str
                The id of the user.
        character_mode : str
            The mode of the character.
        arm_spacing : float
            The spacing between the arms.
        base_x : float
            The x position of the base. (body center)
        base_y : float
            The y position of the base. (height)
        base_z : float
            The z position of the base. (depth)
        """
        if user_id in self._user_arms:
            logger.warning(f"User {user_id} already registered with arm ids {self._user_arms[user_id]}")
            return self._user_arms[user_id]

        mode_spec = get_mode_spec(character_mode)
        arm_count = (
            requested_arm_count
            if mode_spec.factory is None and requested_arm_count and requested_arm_count > 0
            else mode_spec.arm_count
        )
        if mode_spec.base_layout == "octo":
            base_transforms = _allocate_octo_bases(
                user_id,
                base_x=base_x,
                base_y=base_y,
                base_z=base_z,
            )
        else:
            base_transforms = _allocate_linear_bases(
                user_id,
                arm_count=arm_count,
                arm_spacing=arm_spacing,
                base_x=base_x,
                base_y=base_y,
                base_z=base_z,
            )
        allocated_arm_ids = list(base_transforms.keys())

        for arm_id, base in base_transforms.items():
            self._arms[arm_id] = _default_arm_state(arm_id, user_id, base)

        self._user_arms[user_id] = allocated_arm_ids
        self._user_mode[user_id] = character_mode

        factory = mode_spec.factory
        if factory is None:
            return allocated_arm_ids

        if issubclass(factory, DualArmSimulationBase):
            simulation = factory(
                user_id=user_id,
                arm_ids=tuple(allocated_arm_ids),
                base_left=base_transforms[allocated_arm_ids[0]].translation,
                base_right=base_transforms[allocated_arm_ids[1]].translation,
            )
        elif issubclass(factory, OctoArmSimulationBase):
            octo_kwargs: dict[str, object] = {
                "user_id": user_id,
                "arm_ids": tuple(allocated_arm_ids),
                "base_position": (base_x, base_y, base_z),
            }
            if character_mode == "octo-waypoint":
                octo_kwargs["enable_controller_trigger_waypoints"] = True
            simulation = factory(**octo_kwargs)
        else:
            simulation = factory(
                user_id=user_id,
                arm_ids=tuple(allocated_arm_ids),
            )
        self._simulations[user_id] = simulation

        for arm_id in allocated_arm_ids:
            self._previous_commands.pop(arm_id, None)
        self._arms.update(simulation.arm_states())

        # Update other assets
        for mesh in simulation.mesh_entities():
            self.add_or_update_mesh(mesh)
        for sphere in getattr(simulation, "sphere_entities", lambda: [])():
            self.add_or_update_sphere(sphere)

        return allocated_arm_ids

    def remove_user(self, user_id: str) -> None:
        """
        Remove a user from the backend.
        """
        arm_ids = self._user_arms.pop(user_id, [])
        for arm_id in arm_ids:
            self._arms.pop(arm_id, None)
            self._previous_commands.pop(arm_id, None)
        self._user_mode.pop(user_id, None)
        self._simulations.pop(user_id, None)
        self.remove_owner_meshes(user_id)
        self.remove_owner_overlay_points(user_id)
        self.remove_owner_spheres(user_id)

    def step(self, dt: float, command: MultiArmCommand | None) -> SceneState:
        """
        Step the backend.
        """
        self._timestamp += max(0.0, dt)
        if command is not None:
            seen_user_ids: set[str] = set()
            seen_arm_ids = set(command.commands.keys())
            for arm_id, arm_command in command.commands.items():
                if arm_id not in self._arms:
                    continue
                state = self._arms[arm_id]
                if state.owner_user_id:
                    seen_user_ids.add(state.owner_user_id)
                self._apply_command(self._arms[arm_id], arm_command)

            for user_id in seen_user_ids:
                simulation = self._simulations.get(user_id)
                if simulation is not None:
                    simulation.handle_frame_command(command)

            for arm_id in list(self._previous_commands.keys()):
                if arm_id in seen_arm_ids:
                    continue
                state = self._arms.get(arm_id)
                if state is None:
                    continue
                user_id = state.owner_user_id or ""
                simulation = self._simulations.get(user_id)
                if simulation is None:
                    continue
                simulation.handle_command_inactive(arm_id)
                self._previous_commands.pop(arm_id, None)

        # Step the simulations and merge their meshes/spheres into the backend.
        for simulation in self._simulations.values():
            simulation.step(max(dt, 1.0e-4))
            self._arms.update(simulation.arm_states())
            for mesh in simulation.mesh_entities():
                self.add_or_update_mesh(mesh)
            for sphere in getattr(simulation, "sphere_entities", lambda: [])():
                self.add_or_update_sphere(sphere)

        return SceneState(
            timestamp=self._timestamp,
            arms=self._arms,
            user_arms=self._user_arms,
            meshes=self._meshes,
            overlay_points=self._overlay_points,
            spheres=self._spheres,
        )

    def add_or_update_mesh(self, mesh: MeshEntity) -> None:
        """
        Add or update a mesh.
        """
        self._meshes[mesh.mesh_id] = mesh

    def remove_mesh(self, mesh_id: str, owner_id: str | None = None) -> None:
        """
        Remove a mesh.
        """
        mesh = self._meshes.get(mesh_id)
        if mesh is None:
            return
        if owner_id is not None and mesh.owner_id != owner_id:
            return
        self._meshes.pop(mesh_id, None)

    def remove_owner_meshes(self, owner_id: str) -> None:
        """
        Remove all meshes owned by a user.
        """
        mesh_ids = [
            mesh_id
            for mesh_id, mesh in self._meshes.items()
            if mesh.owner_id == owner_id
        ]
        for mesh_id in mesh_ids:
            self._meshes.pop(mesh_id, None)

    def update_mesh_transform(
        self,
        mesh_id: str,
        owner_id: str,
        translation: list[float] | None = None,
        rotation_xyzw: list[float] | None = None,
        scale: list[float] | None = None,
        visible: bool | None = None,
    ) -> bool:
        """
        Update the transform of a mesh.
        Temporary method for testing.
        """
        mesh = self._meshes.get(mesh_id)
        if mesh is None or mesh.owner_id != owner_id:
            return False
        if translation is not None:
            mesh.translation = translation
        if rotation_xyzw is not None:
            mesh.rotation_xyzw = rotation_xyzw
        if scale is not None:
            mesh.scale = scale
        if visible is not None:
            mesh.visible = visible
        return True

    def add_or_update_sphere(self, sphere: SphereEntity) -> None:
        """
        Add or update a sphere.
        """
        self._spheres[sphere.sphere_id] = sphere

    def remove_owner_spheres(self, owner_id: str) -> None:
        """
        Remove all spheres owned by a user.
        """
        sphere_ids = [
            sphere_id
            for sphere_id, sphere in self._spheres.items()
            if sphere.owner_id == owner_id
        ]
        for sphere_id in sphere_ids:
            self._spheres.pop(sphere_id, None)

    def add_or_update_overlay_points(self, overlay: OverlayPointsEntity) -> None:
        """
        Add or update an overlay points.
        Used for point-cloud style dots.
        """
        self._overlay_points[overlay.overlay_id] = overlay


    def remove_overlay_points(
        self, overlay_id: str, owner_id: str | None = None
    ) -> None:
        """
        Remove an overlay points.
        """
        overlay = self._overlay_points.get(overlay_id)
        if overlay is None:
            return
        if owner_id is not None and overlay.owner_id != owner_id:
            return
        self._overlay_points.pop(overlay_id, None)

    def remove_owner_overlay_points(self, owner_id: str) -> None:
        """
        Remove all overlay points owned by a user.
        """
        overlay_ids = [
            overlay_id
            for overlay_id, overlay in self._overlay_points.items()
            if overlay.owner_id == owner_id
        ]
        for overlay_id in overlay_ids:
            self._overlay_points.pop(overlay_id, None)

    def _apply_command(self, state: ArmState, command: ArmCommand) -> None:
        user_id = state.owner_user_id or ""
        simulation = self._simulations.get(user_id)
        if simulation is not None:
            previous_command = self._previous_commands.get(state.arm_id)
            simulation.handle_commands(
                state.arm_id,
                command,
                previous_controller_command=previous_command,
            )
            self._previous_commands[state.arm_id] = command
            return

        if not command.active:
            return

        # Default behavior: update the arm state to follow the controller target.
        target = command.target.translation
        base = state.base.translation
        state.tip = Transform(
            translation=[
                target[0],
                target[1],
                max(base[2] + 0.1, target[2]),
            ],
            rotation_xyzw=command.target.rotation_xyzw,
        )
        state.centerline = [
            [base[0], base[1], base[2]],
            [
                (base[0] + state.tip.translation[0]) / 2.0,
                (base[1] + state.tip.translation[1]) / 2.0,
                (base[2] + state.tip.translation[2]) / 2.0,
            ],
            [
                state.tip.translation[0],
                state.tip.translation[1],
                state.tip.translation[2],
            ],
        ]
        state.element_lengths = [
            (
                (state.centerline[i + 1][0] - state.centerline[i][0]) ** 2
                + (state.centerline[i + 1][1] - state.centerline[i][1]) ** 2
                + (state.centerline[i + 1][2] - state.centerline[i][2]) ** 2
            )
            ** 0.5
            for i in range(len(state.centerline) - 1)
        ]
