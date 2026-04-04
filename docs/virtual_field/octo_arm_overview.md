(vf-octo-arm-overview)=
# Octo Arm Overview

> Sill working on this...

`OctoArmSimulationBase` is the runtime base class for character modes that use a
fixed eight-arm layout.

At the moment this path is still under construction. The goal of this page is
just to give future contributors a quick mental model before they read the code.

The base class lives in `src/virtual_field/runtime/mode_base.py`:

```python
from virtual_field.runtime.mode_base import OctoArmSimulationBase
```

Today, the main user of this base class is `CathyForagingSimulation`.

## What this base class provides

`OctoArmSimulationBase` extends the common runtime `SimulationBase` contract and
specializes it for exactly eight arms.

It currently helps with:

- storing the ordered `arm_ids`
- assigning each arm id to one entry in `base_positions`
- exposing convenience helpers such as `rod_for_index()` and `target_for_index()`
- keeping the mode aligned with the rest of the runtime publishing pipeline

The key requirement is that a derived simulation must provide eight base
positions, one for each arm.

## Typical shape

The pattern is similar to the dual-arm base, but instead of left/right-specific
helpers, the octo version is index-based.

```python
from dataclasses import dataclass
from virtual_field.runtime.mode_base import OctoArmSimulationBase

@dataclass(slots=True)
class MyOctoSimulation(OctoArmSimulationBase):

    def build_simulation(self) -> None:
        ...
```

After configuration, the base class maps:

- `arm_ids[0]` to `base_positions[0]`
- `arm_ids[1]` to `base_positions[1]`
- ...
- `arm_ids[7]` to `base_positions[7]`

## Current status

This area is still evolving.

In particular:

- the octo-arm mode surface is newer than the dual-arm path
- documentation is intentionally light for now
- some mode-specific initialization details are still being refined

So if you are extending this system, use this page as orientation only, then
read:

- `src/virtual_field/runtime/mode_base.py`
- `src/virtual_field/runtime/cathy_foraging_simulation.py`
- `src/virtual_field/runtime/mode_registry.py`

## Notes for future work

Likely follow-up documentation topics:

- expected `build_simulation()` responsibilities for octo-arm modes
- recommended base-layout conventions
- how octo-arm modes publish `ArmState`, `SphereEntity`, and optional meshes
- how frontend rendering should treat multi-arm characters

For now, this page should be treated as an overview stub rather than a complete
developer guide.
