from __future__ import annotations

from collections.abc import Callable

from virtual_field.runtime.mode_base import SimulationBase
from virtual_field.runtime.cathy_throw_simulation import CathyThrowSimulation
from virtual_field.runtime.two_cr_simulation import TwoCRSimulation

SimulationFactory = Callable[..., object]

# Add new mode here:
SIMULATION_FACTORIES: dict[str, SimulationFactory] = {
    "two-cr": TwoCRSimulation,
    "cathy-throw": CathyThrowSimulation,
}

DEFAULT_CHARACTER_MODE = "demo-spline"
SUPPORTED_CHARACTER_MODES = frozenset(
    [DEFAULT_CHARACTER_MODE, *SIMULATION_FACTORIES.keys()]
)
