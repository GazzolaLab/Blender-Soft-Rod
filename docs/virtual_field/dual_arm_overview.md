(vf-dual-arm-overview)=
# Dual Arm Overview

The WebXR runtime modes in `src/virtual_field/runtime/` share a base class
`DualArmSimulationBase` that captures the minimum interface the runtime expects from a dual-arm simulator.

```python
from dataclasses import dataclass, field
from virtual_field.runtime.mode_base import DualArmSimulationBase

@dataclass(slots=True)
class MyModeSimulation(DualArmSimulationBase):

    def build_simulation(self) -> None:
        # (Required) initialize the simulation
        ...

    def post_mode_setup(self) -> None:
        # (Optional) setup after the simulation is built
        ...

```

Once you created the simulation class, you can register it in `src/virtual_field/runtime/mode_registry.py`:

```python
from virtual_field.runtime.my_mode_simulation import MyModeSimulation

SIMULATION_FACTORIES = {
    ...,
    "my-mode": MyModeSimulation,
}
```

If you want to add a new mode, start there and then register the mode in one place.
Once it is registered there, both the backend and websocket hello handling will pick it up through the shared registry.


## Features that are already included in this mode

Do not override `__post_init__` in the derived class for normal mode setup.
`DualArmSimulationBase.__post_init__()` now owns the initialization sequence.

Instead, implement `build_simulation()` and let the base class call it for you.

Inside `build_simulation()`, the base class expects these attributes to be instantiated:

- `self.simulator`: an instance of `elastica.BaseSystemCollection`
- `self.timestepper`: an instance of `elastica.TimeStepper`
- `self.left_rod`: an instance of `elastica.CosseratRod`
- `self.right_rod`: an instance of `elastica.CosseratRod`

```python
    def build_simulation(self) -> None:

        # NOTE: I would recommend importing elastica inside. It is hard to explain why
        # at this point, but it would make the future implementation easier, where multiple
        # simulation needs to run in different threads or SMP is needed.
        import elastica as ea
        class _Simulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
            ea.Contact,
        ):
            pass

        self.simulator = _Simulator()
        self.timestepper = ea.PositionVerlet()
        self.left_rod = ea.CosseratRod.straight_rod( ... )
        self.right_rod = ea.CosseratRod.straight_rod( ... )

        self.simulator.append(self.left_rod)
        self.simulator.append(self.right_rod)
        ...

```

After `build_simulation()` returns, the base class automatically:

- initializes shared target and rest-pose state
- initializes left/right attachment state
- exposes convenience methods such as `get_target_left()` and `is_left_attached()`
- calls `post_mode_setup()` as an optional advanced hook

### Optional advanced hook

Advanced users may override:

```python
def post_mode_setup(self) -> None:
    ...
```

Use this only when you need extra setup after the base class has already created
shared target state. Most modes should implement only `build_simulation()`.

### Data Schema

Rod data schema for VR rendering:

> [TODO] Add data schema for communicating with VR frontend client.

(vf-simple-control-example)=
## Simple Control Example

`DualArmSimulationBase` exposes these convenience methods to obtain controller pose information.
Target position is 3D position (world coordinates), and target orientation is 3x3 row-wise director matrix (local to world). It is `PyElastica` convention.

- `get_target_left()` returns `(target_position, target_orientation)` for the left controller
- `get_target_right()` returns `(target_position, target_orientation)` for the right controller
- `is_left_attached()`
- `is_right_attached()`

Here is example of usage with `TargetPoseProportionalControl`:

```python
self.simulator.add_forcing_to(self.left_rod).using(
    TargetPoseProportionalControl,
    elem_index=-1,
    p_linear_value=200.0,
    p_angular_value=5.0,
    target=self.get_target_left,
    is_attached=self.is_left_attached,
    ramp_up_time=1e-3,
)

self.simulator.add_forcing_to(self.right_rod).using(
    TargetPoseProportionalControl,
    elem_index=-1,
    p_linear_value=200.0,
    p_angular_value=5.0,
    target=self.get_target_right,
    is_attached=self.is_right_attached,
    ramp_up_time=1e-3,
)
```
For more details about tip control forces, see {ref}`target-pose-proportional-control`.

For a basic dual-arm simulation example, see `two_cr_simulation.py` in `virtual_field/runtime`.

## Other features of this base class

- accept target poses from XR controllers (`get_target_left()` and `get_target_right()`)
- accept attachment state from XR controllers (`is_left_attached()` and `is_right_attached()`)
    - By default, it is the `X` and `A` button on the Quest controller.
- controller-orientation recalibration (`Y` and `B` button on the Quest controller)

Behind, it handles:

- advance its internal simulation state in fixed substeps
- expose per-arm `ArmState` snapshots for publishing

There are other features that will be covered in another section:

- optionally publish extra `MeshEntity` or `SphereEntity` assets
- optionally react to extra buttons like sucker or base-pull actions

That means a new mode usually only needs to focus on its physics setup and mode-specific extras.

## (Frontend) Runtime flow

The full path for a mode looks like this:

1. The browser sends `hello` with a `character_mode` string.
2. `VRWebSocketServer` validates that mode against `SUPPORTED_CHARACTER_MODES` from `mode_registry.py`.
3. `MultiArmPassThroughBackend.register_user()` allocates arm ids and creates the simulation object from `SIMULATION_FACTORIES`.
4. Incoming `xr_input` messages are turned into `MultiArmCommand` objects by `SessionArmControlMapper`.
5. `MultiArmPassThroughBackend._apply_command()` forwards controller targets and buttons into your simulation methods.
6. On every simulation tick, the backend calls `simulation.step()`, `simulation.arm_states()`, `simulation.mesh_entities()`, and `simulation.sphere_entities()`.

If you keep your mode inside that contract, the websocket server and VR client do not need mode-specific branching.
