from __future__ import annotations

from typing import Protocol

from .commands import MultiArmCommand, XRInputSample
from .state import SceneState


class SimulationBackend(Protocol):
    def step(
        self, dt: float, command: MultiArmCommand | None
    ) -> SceneState: ...


class ControlMapper(Protocol):
    def map_input(self, sample: XRInputSample) -> MultiArmCommand: ...


class StateTransport(Protocol):
    async def send_state(self, state: SceneState) -> None: ...
