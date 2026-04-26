from __future__ import annotations

from typing import Any

import asyncio
import base64
import json
import ssl
import sys
import time
from dataclasses import dataclass, field
from itertools import count

import click
from loguru import logger
from websockets import WebSocketServerProtocol
from websockets.server import serve

from virtual_field.core.commands import (
    ControllerDisconnectedError,
    MultiArmCommand,
    XRInputSample,
)
from virtual_field.core.mapping import SessionArmControlMapper
from virtual_field.core.state import MeshEntity, OverlayPointsEntity, SceneState
from virtual_field.runtime.mode_registry import SUPPORTED_CHARACTER_MODES

from .backends import MultiArmPassThroughBackend
from .schema import make_message, validate_message
from .teleop import TeleopService


@dataclass(slots=True)
class ClientSession:
    websocket: WebSocketServerProtocol
    user_id: str
    arm_ids: list[str]
    teleop: TeleopService | None
    role: str = "vr_client"
    character_mode: str | None = None
    requested_arm_count: int | None = None
    last_command: MultiArmCommand | None = None
    last_command_ts: float = 0.0
    sent_static_mesh_asset_ids: set[str] = field(default_factory=set)


class VRWebSocketServer:
    """WebSocket entry point for Virtual Field clients and publishers.

    Owns a :class:`~virtual_field.server.backends.MultiArmPassThroughBackend`,
    accepts JSON messages (``hello``, ``xr_input``, ``heartbeat``, ``reset``,
    publisher asset updates), and runs two asyncio loops: simulation stepping at
    ``sim_hz`` and scene snapshots to subscribers at ``publish_hz``. Roles:
    ``vr_client`` (XR input → teleop → backend), ``publisher`` (meshes/overlays),
    and ``spectator`` (receive-only).
    """

    def __init__(
        self,
        *,  # keyword-only arguments (for safety)
        ssl_context: ssl.SSLContext | None,
        host: str = "127.0.0.1",
        port: int = 8765,
        sim_hz: float = 200.0,
        publish_hz: float = 72.0,
    ) -> None:
        self.host = host
        self.port = port
        self.sim_hz = sim_hz
        self.publish_hz = publish_hz
        self.ssl_context = ssl_context

        self.backend = MultiArmPassThroughBackend()

        self._clients: set[WebSocketServerProtocol] = set()
        self._sessions: dict[WebSocketServerProtocol, ClientSession] = {}
        self._client_user_map: dict[WebSocketServerProtocol, str] = {}
        self._heartbeat_timeout = 5.0
        self._server: Any | None = None
        self._publish_task: asyncio.Task[None] | None = None
        self._simulate_task: asyncio.Task[None] | None = None
        self._user_counter = count(1)
        self._publisher_counter = count(1)
        logger.debug(
            "Initialized VRWebSocketServer host={} port={} sim_hz={} publish_hz={}",
            self.host,
            self.port,
            self.sim_hz,
            self.publish_hz,
        )

    async def start(self) -> None:
        """
        Start the server.
        - Creates a new server instance
        - Starts the publish and simulation tasks
        - Waits for the server to start
        """
        logger.debug("Starting websocket server on {}:{}", self.host, self.port)
        self._server = await serve(
            self._handle_client,
            self.host,
            self.port,
            ssl=self.ssl_context,
            max_size=None,
        )
        # Correct port number if changed by server
        self.port = self._server.sockets[0].getsockname()[1]
        self._publish_task = asyncio.create_task(self._publish_loop())
        self._simulate_task = asyncio.create_task(self._simulation_loop())
        self._publish_task.add_done_callback(
            lambda task: self._log_background_task_failure("publish loop", task)
        )
        self._simulate_task.add_done_callback(
            lambda task: self._log_background_task_failure(
                "simulation loop", task
            )
        )
        logger.debug("Server started. bound_port={}", self.port)

    async def stop(self) -> None:
        """
        Stop the server.
        - Cancels the publish and simulation tasks
        - Closes the server
        - Waits for the server to close
        - Closes all client connections
        """
        logger.debug("Stopping server. active_clients={}", len(self._clients))
        if self._publish_task is not None:
            self._publish_task.cancel()
        if self._simulate_task is not None:
            self._simulate_task.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

        for client in tuple(self._clients):
            await client.close()
        logger.debug("Server stopped. closed_clients={}", len(self._clients))

    async def _handle_client(self, websocket: WebSocketServerProtocol) -> None:
        self._clients.add(websocket)
        logger.debug("Client connected. active_clients={}", len(self._clients))
        try:
            async for message in websocket:
                responses = await self._handle_raw_message(websocket, message)
                for response in responses:
                    await websocket.send(json.dumps(response))
        finally:
            self._clients.discard(websocket)
            session = self._sessions.pop(websocket, None)
            if session is not None:
                logger.debug(
                    "Cleaning up session user_id={} role={}",
                    session.user_id,
                    session.role,
                )
                if session.role == "vr_client":
                    self.backend.remove_user(session.user_id)
                self.backend.remove_owner_meshes(session.user_id)
                self.backend.remove_owner_overlay_points(session.user_id)
            self._client_user_map.pop(websocket, None)
            logger.debug(
                "Client disconnected. active_clients={}", len(self._clients)
            )

    async def _handle_raw_message(
        self, websocket: WebSocketServerProtocol, message: str
    ) -> list[dict[str, Any]]:
        try:
            payload = json.loads(message)
            validate_message(payload)
            message_type = payload["type"]
            body = payload["payload"]
            logger.debug("Received message type={}", message_type)

            if message_type == "hello":
                return self._handle_hello(websocket, body)

            session = self._sessions.get(websocket)
            if session is None:
                return [
                    make_message("error", {"reason": "hello required first"})
                ]

            if session.role == "publisher":
                logger.debug(
                    "Routing publisher message type={} owner_id={}",
                    message_type,
                    session.user_id,
                )
                return self._handle_publisher_message(
                    session, message_type, body
                )

            if session.role == "spectator":
                if message_type == "heartbeat":
                    session.last_command_ts = time.monotonic()
                    return []
                return [
                    make_message(
                        "error",
                        {
                            "reason": f"{message_type} unsupported for spectator role"
                        },
                    )
                ]

            if message_type == "heartbeat":
                session.last_command_ts = time.monotonic()
                return []

            if message_type == "reset":
                logger.debug("Reset requested for user_id={}", session.user_id)
                self.backend.remove_user(session.user_id)
                session.arm_ids = self.backend.register_user(
                    session.user_id,
                    character_mode=session.character_mode,
                    requested_arm_count=session.requested_arm_count,
                )
                session.teleop = TeleopService(
                    SessionArmControlMapper(
                        controlled_arm_ids=(
                            session.arm_ids[0],
                            (
                                session.arm_ids[1]
                                if len(session.arm_ids) > 1
                                else session.arm_ids[0]
                            ),
                        )
                    )
                )
                return [
                    make_message(
                        "hello_ack",
                        {
                            "reset": True,
                            "user_id": session.user_id,
                            "arm_ids": session.arm_ids,
                            "controlled_arm_ids": session.arm_ids[:2],
                        },
                    )
                ]

            if message_type == "xr_input":
                if session.teleop is None:
                    return [
                        make_message(
                            "error",
                            {"reason": "xr_input requires vr_client role"},
                        )
                    ]
                sample = XRInputSample.from_dict(body)
                session.last_command = session.teleop.map_input(sample)
                session.last_command_ts = time.monotonic()
                return []

            return [
                make_message(
                    "error", {"reason": f"unsupported type: {message_type}"}
                )
            ]

        except ControllerDisconnectedError:
            return [make_message("error", {"reason": ""})]

        except Exception as exc:  # pragma: no cover - safety fallback
            logger.exception("Failed to handle incoming message: {}", exc)
            return [make_message("error", {"reason": str(exc)})]

    def _handle_hello(
        self, websocket: WebSocketServerProtocol, body: dict[str, Any]
    ) -> list[dict[str, Any]]:
        role = str(body.get("role", "vr_client"))
        logger.debug("Processing hello role={}", role)
        if role == "publisher":
            requested_owner_id = str(body.get("owner_id", "")).strip()
            owner_id = (
                requested_owner_id
                if requested_owner_id
                else f"publisher_{next(self._publisher_counter)}"
            )
            self._client_user_map[websocket] = owner_id
            self._sessions[websocket] = ClientSession(
                websocket=websocket,
                user_id=owner_id,
                arm_ids=[],
                teleop=None,
                role="publisher",
                last_command_ts=time.monotonic(),
            )
            logger.debug("Publisher registered owner_id={}", owner_id)
            return [
                make_message(
                    "hello_ack",
                    {
                        "protocol": 1,
                        "server_time": time.time(),
                        "role": "publisher",
                        "owner_id": owner_id,
                    },
                )
            ]

        if role == "spectator":
            spectator_id = f"spectator_{next(self._user_counter)}"
            self._client_user_map[websocket] = spectator_id
            self._sessions[websocket] = ClientSession(
                websocket=websocket,
                user_id=spectator_id,
                arm_ids=[],
                teleop=None,
                role="spectator",
                last_command_ts=time.monotonic(),
            )
            logger.debug("Spectator registered user_id={}", spectator_id)
            return [
                make_message(
                    "hello_ack",
                    {
                        "protocol": 1,
                        "server_time": time.time(),
                        "role": "spectator",
                        "user_id": spectator_id,
                        "arm_ids": [],
                        "controlled_arm_ids": [],
                    },
                ),
                make_message(
                    "asset_manifest",
                    {
                        "user_id": spectator_id,
                        "arms": {},
                        "scenery": {},
                    },
                ),
            ]

        requested_user_id = str(body.get("user_id", "")).strip()
        requested_mode = body.get("character_mode")
        if (
            not isinstance(requested_mode, str)
            or requested_mode not in SUPPORTED_CHARACTER_MODES
        ):
            return [
                make_message(
                    "error",
                    {
                        "reason": "vr_client requires a backend-supported character_mode"
                    },
                )
            ]
        character_mode = requested_mode
        requested_arm_count_raw = body.get("requested_arm_count")
        try:
            requested_arm_count = (
                max(1, int(requested_arm_count_raw))
                if requested_arm_count_raw is not None
                else None
            )
        except (TypeError, ValueError):
            requested_arm_count = None

        if requested_user_id:
            user_id = requested_user_id
        else:
            user_id = f"user_{next(self._user_counter)}"

        self._client_user_map[websocket] = user_id
        arm_ids = self.backend.register_user(
            user_id,
            character_mode=character_mode,
            requested_arm_count=requested_arm_count,
        )
        logger.debug(
            "VR client registered user_id={} arm_count={} mode={}",
            user_id,
            len(arm_ids),
            character_mode,
        )

        # For now controllers only drive first two arms.
        if len(arm_ids) == 1:
            controlled_arm_ids = (arm_ids[0], arm_ids[0])
        else:
            controlled_arm_ids = (arm_ids[0], arm_ids[1])

        teleop = TeleopService(
            SessionArmControlMapper(controlled_arm_ids=controlled_arm_ids)
        )
        self._sessions[websocket] = ClientSession(
            websocket=websocket,
            user_id=user_id,
            arm_ids=arm_ids,
            teleop=teleop,
            character_mode=character_mode,
            requested_arm_count=requested_arm_count,
            last_command_ts=time.monotonic(),
        )

        responses = [
            make_message(
                "hello_ack",
                {
                    "protocol": 1,
                    "server_time": time.time(),
                    "role": "vr_client",
                    "character_mode": character_mode,
                    "user_id": user_id,
                    "arm_ids": arm_ids,
                    "controlled_arm_ids": list(controlled_arm_ids),
                },
            )
        ]

        manifest_arms: dict[str, dict[str, str]] = {}
        if character_mode == "cathy-foraging":
            for arm_id in arm_ids:
                manifest_arms[arm_id] = {"color": "#8c73fa"}
        else:
            palette = [
                "#ff6b6b",
                "#74c0fc",
                "#8ce99a",
                "#ffd43b",
                "#f783ac",
                "#63e6be",
            ]
            for idx, arm_id in enumerate(arm_ids):
                manifest_arms[arm_id] = {"color": palette[idx % len(palette)]}

        responses.append(
            make_message(
                "asset_manifest",
                {
                    "user_id": user_id,
                    "arms": manifest_arms,
                    "scenery": {},
                },
            )
        )
        return responses

    def _handle_publisher_message(
        self, session: ClientSession, message_type: str, body: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if message_type == "heartbeat":
            session.last_command_ts = time.monotonic()
            return []

        if message_type == "add_mesh":
            mesh_id = str(body.get("mesh_id", "")).strip()
            mesh_data_b64 = str(body.get("mesh_data_b64", "")).strip()
            if not mesh_id or not mesh_data_b64:
                return [
                    make_message(
                        "error",
                        {
                            "reason": "add_mesh requires mesh_id and mesh_data_b64"
                        },
                    )
                ]

            mime_type = str(body.get("mime_type", "model/gltf-binary"))
            base64.b64decode(mesh_data_b64, validate=True)
            asset_uri = f"data:{mime_type};base64,{mesh_data_b64}"

            mesh = MeshEntity(
                mesh_id=mesh_id,
                owner_id=session.user_id,
                asset_uri=asset_uri,
                translation=list(body.get("translation", [0.0, 0.0, 0.0])),
                rotation_xyzw=list(
                    body.get("rotation_xyzw", [0.0, 0.0, 0.0, 1.0])
                ),
                scale=list(body.get("scale", [1.0, 1.0, 1.0])),
                visible=bool(body.get("visible", True)),
                static_asset=bool(body.get("static_asset", False)),
            )
            self.backend.add_or_update_mesh(mesh)
            logger.debug(
                "Mesh added/updated owner_id={} mesh_id={}",
                session.user_id,
                mesh_id,
            )
            return [
                make_message(
                    "mesh_ack",
                    {
                        "owner_id": session.user_id,
                        "mesh_id": mesh_id,
                        "status": "added",
                    },
                )
            ]

        if message_type == "remove_mesh":
            mesh_id = str(body.get("mesh_id", "")).strip()
            self.backend.remove_mesh(mesh_id, owner_id=session.user_id)
            logger.debug(
                "Mesh removed owner_id={} mesh_id={}", session.user_id, mesh_id
            )
            return [
                make_message(
                    "mesh_ack",
                    {
                        "owner_id": session.user_id,
                        "mesh_id": mesh_id,
                        "status": "removed",
                    },
                )
            ]

        if message_type == "update_mesh_transform":
            mesh_id = str(body.get("mesh_id", "")).strip()
            if not mesh_id:
                return [
                    make_message(
                        "error",
                        {"reason": "update_mesh_transform requires mesh_id"},
                    )
                ]
            updated = self.backend.update_mesh_transform(
                mesh_id=mesh_id,
                owner_id=session.user_id,
                translation=(
                    list(body["translation"]) if "translation" in body else None
                ),
                rotation_xyzw=(
                    list(body["rotation_xyzw"])
                    if "rotation_xyzw" in body
                    else None
                ),
                scale=list(body["scale"]) if "scale" in body else None,
                visible=(bool(body["visible"]) if "visible" in body else None),
            )
            if not updated:
                return [
                    make_message(
                        "error",
                        {
                            "reason": "mesh not found or ownership mismatch for update"
                        },
                    )
                ]
            return [
                make_message(
                    "mesh_ack",
                    {
                        "owner_id": session.user_id,
                        "mesh_id": mesh_id,
                        "status": "updated",
                    },
                )
            ]

        if message_type == "clear_meshes":
            self.backend.remove_owner_meshes(session.user_id)
            logger.debug("Meshes cleared owner_id={}", session.user_id)
            return [
                make_message(
                    "mesh_ack",
                    {
                        "owner_id": session.user_id,
                        "status": "cleared",
                    },
                )
            ]

        if message_type == "update_overlay_points":
            overlay_id = str(body.get("overlay_id", "")).strip()
            if not overlay_id:
                return [
                    make_message(
                        "error",
                        {"reason": "update_overlay_points requires overlay_id"},
                    )
                ]
            raw_points = body.get("points", [])
            if not isinstance(raw_points, list):
                return [
                    make_message(
                        "error",
                        {"reason": "points must be a list of [x, y, z]"},
                    )
                ]
            points = [list(point) for point in raw_points]
            overlay = OverlayPointsEntity(
                overlay_id=overlay_id,
                owner_id=session.user_id,
                points=points,
                point_size=float(body.get("point_size", 0.008)),
                visible=bool(body.get("visible", True)),
            )
            self.backend.add_or_update_overlay_points(overlay)
            logger.debug(
                "Overlay points updated owner_id={} overlay_id={} point_count={}",
                session.user_id,
                overlay_id,
                len(points),
            )
            return [
                make_message(
                    "overlay_ack",
                    {
                        "owner_id": session.user_id,
                        "overlay_id": overlay_id,
                        "status": "updated",
                    },
                )
            ]

        if message_type == "remove_overlay_points":
            overlay_id = str(body.get("overlay_id", "")).strip()
            self.backend.remove_overlay_points(
                overlay_id, owner_id=session.user_id
            )
            logger.debug(
                "Overlay points removed owner_id={} overlay_id={}",
                session.user_id,
                overlay_id,
            )
            return [
                make_message(
                    "overlay_ack",
                    {
                        "owner_id": session.user_id,
                        "overlay_id": overlay_id,
                        "status": "removed",
                    },
                )
            ]

        if message_type == "clear_overlay_points":
            self.backend.remove_owner_overlay_points(session.user_id)
            logger.debug("Overlay points cleared owner_id={}", session.user_id)
            return [
                make_message(
                    "overlay_ack",
                    {
                        "owner_id": session.user_id,
                        "status": "cleared",
                    },
                )
            ]

        return [
            make_message(
                "error",
                {
                    "reason": f"unsupported publisher message type: {message_type}"
                },
            )
        ]

    async def _simulation_loop(self) -> None:
        dt = 1.0 / self.sim_hz
        logger.debug("Simulation loop started dt={}", dt)
        while True:
            command: MultiArmCommand | None = None
            vr_client_count = 0
            for session in self._sessions.values():
                if session.role != "vr_client":
                    continue
                vr_client_count += 1
                if session.last_command is None:
                    continue
                if (
                    time.monotonic() - session.last_command_ts
                    > self._heartbeat_timeout
                ):
                    session.last_command = None
                    continue
                if command is None:
                    command = session.last_command
            if vr_client_count:
                # One step per tick for the shared backend. ``command`` may be None
                # until the first ``xr_input``; physics (preset octo waypoints, etc.)
                # still runs — trigger is not required for stepping.
                self.backend.step(dt, command)
            elif not self._sessions:
                self.backend.step(dt, None)
            await asyncio.sleep(dt)

    async def _publish_loop(self) -> None:
        dt = 1.0 / self.publish_hz
        logger.debug("Publish loop started dt={}", dt)
        while True:
            if self._clients:
                state = self.backend.step(0.0, None)
                await self._broadcast_scene_state(state)
            await asyncio.sleep(dt)

    def _log_background_task_failure(
        self, task_name: str, task: asyncio.Task[None]
    ) -> None:
        if task.cancelled():
            logger.debug("{} cancelled", task_name)
            return
        try:
            exc = task.exception()
        except (
            Exception
        ):  # pragma: no cover - defensive task-inspection fallback
            logger.exception("Failed to inspect {}", task_name)
            return
        if exc is not None:
            logger.opt(exception=exc).error("{} crashed", task_name)

    async def _broadcast(self, message: dict[str, Any]) -> None:
        encoded = json.dumps(message)
        stale: list[WebSocketServerProtocol] = []
        for client in tuple(self._clients):
            try:
                await client.send(encoded)
            except (
                Exception
            ):  # pragma: no cover - network transport failure path
                stale.append(client)
        for client in stale:
            self._clients.discard(client)
        if stale:
            logger.debug("Dropped stale clients count={}", len(stale))

    async def _broadcast_scene_state(self, state: SceneState) -> None:
        """Send ``scene_state`` with per-client omission of static mesh ``asset_uri``."""
        stale: list[WebSocketServerProtocol] = []
        for client in tuple(self._clients):
            session = self._sessions.get(client)
            if session is None:
                payload = state.to_dict()
            else:
                payload = state.to_dict_for_client(
                    session.sent_static_mesh_asset_ids
                )
            message = make_message("scene_state", payload)
            encoded = json.dumps(message)
            try:
                await client.send(encoded)
            except (
                Exception
            ):  # pragma: no cover - network transport failure path
                stale.append(client)
        for client in stale:
            self._clients.discard(client)
        if stale:
            logger.debug("Dropped stale clients count={}", len(stale))


async def run_server(
    host: str, port: int, ssl_context: ssl.SSLContext | None
) -> None:
    server = VRWebSocketServer(host=host, port=port, ssl_context=ssl_context)
    await server.start()

    scheme = "wss" if ssl_context is not None else "ws"
    logger.info(
        "VR server listening on {}://{}:{}",
        scheme,
        server.host,
        server.port,
    )

    # Run the server until interrupted.
    try:
        while True:
            await asyncio.sleep(3600)
    except (
        Exception
    ) as exc:  # pragma: no cover - long-running server loop guard
        logger.error("Exception in run_server: {}", exc)
    finally:
        logger.info("Stopping server")
        await server.stop()


def configure_logging(verbose: bool) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "INFO")


@click.command(help="Run Virtual Field VR runtime server")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", type=int, default=8765, show_default=True)
@click.option("--ssl-cert", type=click.Path(exists=True, dir_okay=False))
@click.option("--ssl-key", type=click.Path(exists=True, dir_okay=False))
@click.option("--verbose", is_flag=True, help="Enable debug logging output.")
def main(
    host: str,
    port: int,
    ssl_cert: str | None,
    ssl_key: str | None,
    verbose: bool,
) -> None:
    configure_logging(verbose=verbose)

    # Validate and configure optional TLS.
    if (ssl_cert is None) != (ssl_key is None):
        raise click.UsageError(
            "--ssl-cert and --ssl-key must be provided together"
        )

    ssl_context: ssl.SSLContext | None = None
    if ssl_cert is not None and ssl_key is not None:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(ssl_cert, ssl_key)

    # Run the server.
    asyncio.run(
        run_server(
            host=host,
            port=port,
            ssl_context=ssl_context,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
