from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from virtual_field.runtime.cathy_foraging_simulation import (
    CathyForagingSimulation,
)
from virtual_field.runtime.cathy_throw_simulation import CathyThrowSimulation
from virtual_field.runtime.coomm_octopus_simulation import (
    COOMMOctopusSimulation,
)
from virtual_field.runtime.mode_base import SimulationBase
from virtual_field.runtime.noel_c4_simulation import NoelC4Simulation
from virtual_field.runtime.octo_waypoint_simulation import (
    OctoWaypointSimulation,
)
from virtual_field.runtime.spirobs_simulation import SpirobsSimulation
from virtual_field.runtime.two_cr_simulation import TwoCRSimulation
from virtual_field.runtime.two_gcr_simulation import TwoGCRSimulation

SimulationFactory = Callable[..., SimulationBase]


@dataclass(frozen=True, slots=True)
class CharacterModeSpec:
    """Describes the configuration for a character mode.

    Attributes
    ----------
    arm_count : int
        Number of arms (effectors) the character has in this mode.
    base_layout : str
        Base layout style; typically 'linear' or 'octo'.
    factory : SimulationFactory
        Optional class that produces a SimulationBase instance for this mode.
    """

    arm_count: int
    base_layout: str
    factory: SimulationFactory


MODE_SPECS: dict[str, CharacterModeSpec] = {
    "two-cr": CharacterModeSpec(
        arm_count=2, base_layout="linear", factory=TwoCRSimulation
    ),
    "two-gcr": CharacterModeSpec(
        arm_count=2, base_layout="linear", factory=TwoGCRSimulation
    ),
    "spirobs": CharacterModeSpec(
        arm_count=2, base_layout="linear", factory=SpirobsSimulation
    ),
    "cathy-throw": CharacterModeSpec(
        arm_count=2, base_layout="linear", factory=CathyThrowSimulation
    ),
    "coomm-octopus": CharacterModeSpec(
        arm_count=2, base_layout="linear", factory=COOMMOctopusSimulation
    ),
    "noel-c4": CharacterModeSpec(
        arm_count=2, base_layout="linear", factory=NoelC4Simulation
    ),
    "cathy-foraging": CharacterModeSpec(
        arm_count=8, base_layout="octo", factory=CathyForagingSimulation
    ),
    "octo-waypoint": CharacterModeSpec(
        arm_count=9, base_layout="octo", factory=OctoWaypointSimulation
    ),
}
SUPPORTED_CHARACTER_MODES = frozenset(MODE_SPECS.keys())


def get_mode_spec(character_mode: str) -> CharacterModeSpec:
    return MODE_SPECS.get(character_mode)
