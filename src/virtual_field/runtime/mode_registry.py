from __future__ import annotations

from collections.abc import Callable

from virtual_field.runtime.two_cr_simulation import TwoCRSimulation

SimulationFactory = Callable[..., object]

# Add new mode here:
SIMULATION_FACTORIES: dict[str, SimulationFactory] = {
    "two-cr": TwoCRSimulation,
}

DEFAULT_CHARACTER_MODE = "demo-spline"
SUPPORTED_CHARACTER_MODES = frozenset(
    [DEFAULT_CHARACTER_MODE, *SIMULATION_FACTORIES.keys()]
)
