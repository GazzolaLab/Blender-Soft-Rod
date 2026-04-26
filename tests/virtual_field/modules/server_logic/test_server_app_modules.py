import asyncio

import pytest

from virtual_field.server.app import ClientSession, VRWebSocketServer

pytestmark = pytest.mark.modules


def _server() -> VRWebSocketServer:
    return VRWebSocketServer(
        ssl_context=None,  # type: ignore[arg-type]
        host="127.0.0.1",
        port=0,
        sim_hz=120.0,
        publish_hz=30.0,
    )


def test_raw_message_requires_hello_first() -> None:
    server = _server()
    websocket = object()  # type: ignore[assignment]
    responses = asyncio.run(
        server._handle_raw_message(
            websocket, '{"version": 1, "type": "reset", "payload": {}}'
        )
    )
    assert responses[0]["type"] == "error"
    assert "hello required first" in responses[0]["payload"]["reason"]


def test_raw_message_rejects_spectator_xr_input() -> None:
    server = _server()
    websocket = object()  # type: ignore[assignment]
    server._sessions[websocket] = ClientSession(
        websocket=websocket,  # type: ignore[arg-type]
        user_id="spectator_1",
        arm_ids=[],
        teleop=None,
        role="spectator",
    )
    responses = asyncio.run(
        server._handle_raw_message(
            websocket,
            '{"version": 1, "type": "xr_input", "payload": {"timestamp": 0, "head_pose": {"translation": [0,0,0], "rotation_xyzw": [0,0,0,1]}, "controllers": {"left": {"pose": {"translation": [0,0,0], "rotation_xyzw": [0,0,0,1]}}}}}',
        )
    )
    assert responses[0]["type"] == "error"
    assert "unsupported for spectator role" in responses[0]["payload"]["reason"]


def test_hello_registers_spectator_with_empty_control() -> None:
    server = _server()
    responses = server._handle_hello(
        object(),  # type: ignore[arg-type]
        {"client": "viewer", "role": "spectator"},
    )
    assert responses[0]["type"] == "hello_ack"
    assert responses[0]["payload"]["role"] == "spectator"
    assert responses[0]["payload"]["controlled_arm_ids"] == []
    assert responses[1]["type"] == "asset_manifest"


def test_publisher_message_add_mesh_requires_fields() -> None:
    server = _server()
    session = ClientSession(
        websocket=object(),  # type: ignore[arg-type]
        user_id="publisher_a",
        arm_ids=[],
        teleop=None,
        role="publisher",
    )
    responses = server._handle_publisher_message(session, "add_mesh", {})
    assert responses[0]["type"] == "error"
    assert "add_mesh requires mesh_id and mesh_data_b64" in responses[0]["payload"]["reason"]


def test_publisher_message_rejects_update_without_mesh_id() -> None:
    server = _server()
    session = ClientSession(
        websocket=object(),  # type: ignore[arg-type]
        user_id="publisher_a",
        arm_ids=[],
        teleop=None,
        role="publisher",
    )
    responses = server._handle_publisher_message(
        session, "update_mesh_transform", {"translation": [0.0, 0.0, 0.0]}
    )
    assert responses[0]["type"] == "error"
    assert "requires mesh_id" in responses[0]["payload"]["reason"]
