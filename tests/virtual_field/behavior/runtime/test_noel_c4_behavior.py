from collections.abc import Callable

import numpy as np
import pytest

from virtual_field.core.commands import ArmCommand, MultiArmCommand
from virtual_field.core.state import Transform
from virtual_field.runtime.cathy_foraging_simulation import (
    CathyForagingSimulation,
)
from virtual_field.runtime.cathy_throw_simulation import CathyThrowSimulation
from virtual_field.runtime.octo_waypoint_simulation import (
    WAYPOINT_PLANE_Y,
    OctoWaypointSimulation,
)
from virtual_field.server.app import VRWebSocketServer
from virtual_field.server.backends import MultiArmPassThroughBackend

pytestmark = pytest.mark.behavior


def test_backend_registers_and_cleans_noel_c4_meshes() -> None:
    backend = MultiArmPassThroughBackend()
    arm_ids = backend.register_user("user_noel", character_mode="noel-c4")
    assert len(arm_ids) == 2
    obstacle_mesh_ids = [
        mesh_id
        for mesh_id in backend._meshes
        if mesh_id.startswith("user_noel_noel_c4_obstacle_")
    ]
    assert len(obstacle_mesh_ids) == 12
    for mesh_id in obstacle_mesh_ids:
        mesh = backend._meshes[mesh_id]
        assert mesh.owner_id == "user_noel"
        assert mesh.asset_uri.startswith("data:model/gltf+json;base64,")

    backend.remove_user("user_noel")
    assert not any(
        mesh.owner_id == "user_noel" for mesh in backend._meshes.values()
    )


def test_existing_dual_modes_still_register() -> None:
    for mode in ("two-cr", "spirobs", "cathy-throw", "coomm-octopus"):
        backend = MultiArmPassThroughBackend()
        arm_ids = backend.register_user(f"user_{mode}", character_mode=mode)
        assert len(arm_ids) == 2
        assert set(arm_ids).issubset(backend._arms.keys())


def test_cathy_foraging_registers_eight_arms() -> None:
    backend = MultiArmPassThroughBackend()
    arm_ids = backend.register_user(
        "user_foraging", character_mode="cathy-foraging"
    )
    assert len(arm_ids) == 8
    assert set(arm_ids).issubset(backend._arms.keys())
    assert len(backend._spheres) == 8


def test_octo_waypoint_registers_nine_arms() -> None:
    backend = MultiArmPassThroughBackend()
    arm_ids = backend.register_user(
        "user_waypoint", character_mode="octo-waypoint"
    )
    assert len(arm_ids) == 9
    assert set(arm_ids).issubset(backend._arms.keys())
    assert "user_waypoint_waypoint_0" in backend._spheres


def test_octo_waypoint_projects_right_controller_to_floor(
    controller_transform: Callable[[list[float], list[float]], Transform],
) -> None:
    simulation = OctoWaypointSimulation(
        user_id="user_waypoint",
        arm_ids=tuple(f"user_waypoint_arm_{idx}" for idx in range(9)),
        base_position=(0.0, 1.0, -0.15),
        seed_pentagon_waypoints=False,
        enable_controller_trigger_waypoints=True,
    )
    right_arm_id = simulation.arm_ids[1]
    simulation.handle_frame_command(
        MultiArmCommand(
            timestamp=0.0,
            commands={
                right_arm_id: ArmCommand(
                    arm_id=right_arm_id,
                    active=True,
                    target=controller_transform(
                        [0.0, 1.5, 0.5], [0.0, 0.0, 0.0, 1.0]
                    ),
                    buttons={"trigger_click": True},
                )
            },
        )
    )
    assert len(simulation._waypoint_queue) == 1
    assert np.isclose(simulation._waypoint_queue[0][1], WAYPOINT_PLANE_Y)
    assert np.allclose(
        simulation._waypoint_queue[0][[0, 2]], [0.0, 0.5], atol=1.0e-6
    )


def test_octo_waypoint_ignores_relocation_outside_visible_plane(
    controller_transform: Callable[[list[float], list[float]], Transform],
) -> None:
    simulation = OctoWaypointSimulation(
        user_id="user_waypoint",
        arm_ids=tuple(f"user_waypoint_arm_{idx}" for idx in range(9)),
        base_position=(0.0, 1.0, -0.15),
        seed_pentagon_waypoints=False,
        enable_controller_trigger_waypoints=True,
    )
    right_arm_id = simulation.arm_ids[1]
    original_waypoint = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    simulation._waypoint_queue = [original_waypoint.copy()]
    simulation.handle_frame_command(
        MultiArmCommand(
            timestamp=0.0,
            commands={
                right_arm_id: ArmCommand(
                    arm_id=right_arm_id,
                    active=True,
                    target=controller_transform(
                        [3.5, 1.5, 0.5], [0.0, 0.0, 0.0, 1.0]
                    ),
                    buttons={"trigger_click": True},
                )
            },
        )
    )
    assert len(simulation._waypoint_queue) == 1
    assert np.allclose(simulation._waypoint_queue[0], original_waypoint)


def test_simulation_backend_applies_controller_command_to_arm_state() -> None:
    backend = MultiArmPassThroughBackend()
    arm_ids = backend.register_user("user_two_cr", character_mode="two-cr")
    arm_id = arm_ids[0]
    before = (
        backend._simulations["user_two_cr"].arm_states()[arm_id].tip.translation
    )
    backend._apply_command(
        backend._arms[arm_id],
        ArmCommand(
            arm_id=arm_id,
            active=True,
            target=Transform(
                translation=[0.30, 1.20, -0.80],
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            buttons={},
        ),
    )
    backend.step(1.0 / 120.0, command=None)
    after = (
        backend._simulations["user_two_cr"].arm_states()[arm_id].tip.translation
    )
    assert not np.allclose(before, after)


def test_cathy_throw_uses_two_spirobs_without_contact_visualization() -> None:
    simulation = CathyThrowSimulation(
        user_id="user_cathy",
        arm_ids=("left_arm", "right_arm"),
        base_left=[-0.15, 1.0, -0.15],
        base_right=[0.15, 1.0, -0.15],
    )
    assert (
        np.ptp(np.asarray(simulation.left_rod.radius, dtype=np.float64)) > 0.0
    )
    assert (
        np.ptp(np.asarray(simulation.right_rod.radius, dtype=np.float64)) > 0.0
    )
    arm_states = simulation.arm_states()
    assert arm_states["left_arm"].contact_points == []
    assert arm_states["right_arm"].contact_points == []


def test_server_accepts_noel_c4_mode() -> None:
    server = VRWebSocketServer(
        ssl_context=None,  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        sim_hz=120.0,
        publish_hz=30.0,
    )
    responses = server._handle_hello(
        object(),  # type: ignore[arg-type]
        {
            "client": "pytest",
            "requested_arm_count": 4,
            "character_mode": "noel-c4",
        },
    )
    hello_ack = responses[0]
    assert hello_ack["type"] == "hello_ack"
    assert hello_ack["payload"]["character_mode"] == "noel-c4"
    assert len(hello_ack["payload"]["arm_ids"]) == 2


def test_backend_step_noel_c4_does_not_raise() -> None:
    backend = MultiArmPassThroughBackend()
    backend.register_user("user_noel_step", character_mode="noel-c4")
    scene_state = backend.step(1.0 / 120.0, command=None)
    assert len(scene_state.arms) == 2


def test_server_accepts_cathy_foraging_mode() -> None:
    server = VRWebSocketServer(
        ssl_context=None,  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        sim_hz=120.0,
        publish_hz=30.0,
    )
    responses = server._handle_hello(
        object(),  # type: ignore[arg-type]
        {
            "client": "pytest",
            "requested_arm_count": 2,
            "character_mode": "cathy-foraging",
        },
    )
    hello_ack = responses[0]
    assert hello_ack["payload"]["character_mode"] == "cathy-foraging"
    assert len(hello_ack["payload"]["arm_ids"]) == 8


def test_cathy_foraging_head_tilt_locks_heading_within_cycle(
    quat_from_euler_xyz: Callable[[float, float, float], list[float]],
) -> None:
    simulation = CathyForagingSimulation(
        user_id="user_cathy",
        arm_ids=tuple(f"arm_{index}" for index in range(8)),
        base_position=(0.0, 1.0, -0.15),
    )
    simulation.handle_frame_command(
        MultiArmCommand(
            timestamp=0.0,
            commands={},
            head_pose=Transform(
                translation=[0.0, 1.4, 0.0],
                rotation_xyzw=quat_from_euler_xyz(-0.35, 0.0, 0.0),
            ),
        )
    )
    simulation._apply_policy()
    locked_heading = simulation._cycle_heading_angle
    simulation.handle_frame_command(
        MultiArmCommand(
            timestamp=0.1,
            commands={},
            head_pose=Transform(
                translation=[0.0, 1.4, 0.0],
                rotation_xyzw=quat_from_euler_xyz(-0.35, 0.0, 0.8),
            ),
        )
    )
    simulation._time = 0.5 * simulation.base_policy.T_L
    simulation._apply_policy()
    assert np.isclose(simulation._cycle_heading_angle, locked_heading)


def test_backend_publishes_cathy_throw_sphere_in_scene_state() -> None:
    backend = MultiArmPassThroughBackend()
    backend.register_user("user_cathy", character_mode="cathy-throw")
    scene_state = backend.step(1.0 / 120.0, command=None)
    sphere = next(iter(scene_state.spheres.values()))
    assert sphere.owner_id == "user_cathy"
    assert sphere.radius == np.float64(0.08)


def test_cathy_throw_trigger_click_enables_sucker_on_matching_arm() -> None:
    backend = MultiArmPassThroughBackend()
    arm_ids = backend.register_user("user_cathy", character_mode="cathy-throw")
    simulation = backend._simulations["user_cathy"]
    backend._apply_command(
        backend._arms[arm_ids[1]],
        ArmCommand(
            arm_id=arm_ids[1],
            active=True,
            target=Transform(
                translation=backend._arms[arm_ids[1]].tip.translation,
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            buttons={"trigger_click": True},
        ),
    )
    assert simulation._sucker_active[arm_ids[0]] is False
    assert simulation._sucker_active[arm_ids[1]] is True
