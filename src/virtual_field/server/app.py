from __future__ import annotations

import asyncio
import base64
import json
import ssl
import sys
import time
from dataclasses import dataclass
from itertools import count
from typing import Any

import click
from loguru import logger
from websockets import WebSocketServerProtocol
from websockets.server import serve

from virtual_field.core.commands import MultiArmCommand, XRInputSample
from virtual_field.core.mapping import SessionArmControlMapper
from virtual_field.core.state import MeshEntity, OverlayPointsEntity
from virtual_field.runtime.mode_registry import (
    DEFAULT_CHARACTER_MODE,
    SUPPORTED_CHARACTER_MODES,
)
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
    character_mode: str = DEFAULT_CHARACTER_MODE
    last_command: MultiArmCommand | None = None
    last_command_ts: float = 0.0


class VRWebSocketServer:
    """TLS WebSocket entry point for Virtual Field clients and publishers.

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
        ssl_context: ssl.SSLContext,
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
        self.port = self._server.sockets[0].getsockname()[1]
        self._publish_task = asyncio.create_task(self._publish_loop())
        self._simulate_task = asyncio.create_task(self._simulation_loop())
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

        for client in self._clients:
            await client.close()
        logger.debug("Server stopped. closed_clients={}", len(self._clients))


async def run_server(
    host: str, port: int, ssl_context: ssl.SSLContext
) -> None:
    server = VRWebSocketServer(host=host, port=port, ssl_context=ssl_context)
    await server.start()

    scheme = "wss"  # always use WSS. Required for Meta Quest API.
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
    finally:
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
    ssl_cert: str,
    ssl_key: str,
    verbose: bool,
) -> None:
    configure_logging(verbose=verbose)

    # Validate that both SSL cert and key are set together.
    ssl_context: ssl.SSLContext
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
