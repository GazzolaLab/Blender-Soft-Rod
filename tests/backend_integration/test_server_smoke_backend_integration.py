import asyncio
import base64
import json

import pytest
from websockets import connect

from virtual_field.server.app import VRWebSocketServer

pytestmark = [pytest.mark.backend_integration, pytest.mark.slow]


async def _recv_message_type(websocket, expected_type: str) -> dict:
    while True:
        message = json.loads(await websocket.recv())
        if message["type"] == expected_type:
            return message


async def _exercise_handshake_smoke() -> None:
    server = VRWebSocketServer(
        host="127.0.0.1",
        port=0,
        sim_hz=120.0,
        publish_hz=30.0,
        ssl_context=None,
    )
    await server.start()
    try:
        uri = f"ws://127.0.0.1:{server.port}"
        async with connect(uri) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "version": 1,
                        "type": "hello",
                        "payload": {
                            "client": "pytest",
                            "character_mode": "noel-c4",
                        },
                    }
                )
            )
            hello_ack = await _recv_message_type(websocket, "hello_ack")
            assert hello_ack["payload"]["character_mode"] == "noel-c4"
            assert len(hello_ack["payload"]["arm_ids"]) == 2
            assert hello_ack["version"] == 1
            _ = await _recv_message_type(websocket, "asset_manifest")
    finally:
        await server.stop()


def test_server_handshake_smoke_backend_integration() -> None:
    asyncio.run(_exercise_handshake_smoke())


async def _exercise_multi_user_allocation_smoke() -> None:
    server = VRWebSocketServer(
        host="127.0.0.1",
        port=0,
        sim_hz=120.0,
        publish_hz=30.0,
        ssl_context=None,
    )
    await server.start()
    try:
        uri = f"ws://127.0.0.1:{server.port}"
        async with connect(uri) as ws_a, connect(uri) as ws_b:
            await ws_a.send(
                json.dumps(
                    {
                        "version": 1,
                        "type": "hello",
                        "payload": {"client": "a", "character_mode": "noel-c4"},
                    }
                )
            )
            hello_a = await _recv_message_type(ws_a, "hello_ack")
            _ = await _recv_message_type(ws_a, "asset_manifest")

            await ws_b.send(
                json.dumps(
                    {
                        "version": 1,
                        "type": "hello",
                        "payload": {
                            "client": "b",
                            "character_mode": "cathy-foraging",
                        },
                    }
                )
            )
            hello_b = await _recv_message_type(ws_b, "hello_ack")
            _ = await _recv_message_type(ws_b, "asset_manifest")

            assert (
                hello_a["payload"]["user_id"] != hello_b["payload"]["user_id"]
            )
            assert len(hello_a["payload"]["arm_ids"]) == 2
            assert len(hello_b["payload"]["arm_ids"]) == 8
    finally:
        await server.stop()


def test_multi_user_allocation_smoke_backend_integration() -> None:
    asyncio.run(_exercise_multi_user_allocation_smoke())


async def _exercise_publisher_mesh_smoke() -> None:
    server = VRWebSocketServer(
        host="127.0.0.1",
        port=0,
        sim_hz=120.0,
        publish_hz=30.0,
        ssl_context=None,
    )
    await server.start()
    try:
        uri = f"ws://127.0.0.1:{server.port}"
        async with connect(uri) as vr_client, connect(uri) as publisher:
            await vr_client.send(
                json.dumps(
                    {
                        "version": 1,
                        "type": "hello",
                        "payload": {
                            "client": "vr",
                            "character_mode": "noel-c4",
                        },
                    }
                )
            )
            _ = await _recv_message_type(vr_client, "hello_ack")
            _ = await _recv_message_type(vr_client, "asset_manifest")

            await publisher.send(
                json.dumps(
                    {
                        "version": 1,
                        "type": "hello",
                        "payload": {
                            "role": "publisher",
                            "owner_id": "pub_smoke",
                        },
                    }
                )
            )
            hello_ack = await _recv_message_type(publisher, "hello_ack")
            assert hello_ack["payload"]["role"] == "publisher"

            payload = base64.b64encode(b"glTF").decode("ascii")
            await publisher.send(
                json.dumps(
                    {
                        "version": 1,
                        "type": "add_mesh",
                        "payload": {
                            "mesh_id": "tree_smoke",
                            "mime_type": "model/gltf-binary",
                            "mesh_data_b64": payload,
                            "translation": [0.0, 0.0, 0.0],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                            "scale": [1.0, 1.0, 1.0],
                        },
                    }
                )
            )
            mesh_ack = await _recv_message_type(publisher, "mesh_ack")
            assert mesh_ack["payload"]["status"] == "added"
            assert mesh_ack["version"] == 1
    finally:
        await server.stop()


def test_publisher_mesh_lifecycle_smoke_backend_integration() -> None:
    asyncio.run(_exercise_publisher_mesh_smoke())
