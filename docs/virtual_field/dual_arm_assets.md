# Publishing Arm State and Assets

This page covers how a mode publishes arm geometry and extra scene objects back to the client.

## Publishing arm state

By default, `DualArmSimulationBase.arm_states()` publishes one `ArmState` for `left_rod` and one for `right_rod`.

The base implementation extracts:

- `centerline` from `rod.position_collection`
- `radii` from `rod.radius`
- `element_lengths` from `rod.lengths` when available
- `directors` from `rod.director_collection`
- `tip.rotation_xyzw` from the final director frame
- `contact_points` from `contact_points_for_arm()`

For many modes that is enough. If your rod state already lives in a standard PyElastica-like structure, you often do not need to override `arm_states()` at all.

## Adding contact-point trails

If you only need extra visualization points, override `contact_points_for_arm()`:

```python
def contact_points_for_arm(self, arm_id: str) -> list[list[float]]:
    queue = self._recording_queues.get(arm_id)
    if queue is None:
        return []
    return [point for _, point in queue]
```

This keeps the default `ArmState` conversion while adding mode-specific point data.

## Customizing `ArmState`

If you need a different arm representation entirely, override `arm_states()` or `_rod_to_arm_state()` and return your own `ArmState` objects.

```python
from virtual_field.core.state import ArmState, Transform

def arm_states(self) -> dict[str, ArmState]:
    return {
        self.arm_ids[0]: ArmState(
            arm_id=self.arm_ids[0],
            owner_user_id=self.user_id,
            base=Transform(
                translation=self.base_left,
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            tip=Transform(
                translation=[0.0, 1.0, -0.5],
                rotation_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
            centerline=[[0.0, 1.0, 0.0], [0.0, 1.0, -0.5]],
            radii=[0.03],
        ),
        self.arm_ids[1]: ...,
    }
```

Important note: the current backend treats registered simulation modes as dual-arm modes and forces `arm_count = 2` during registration. If you want a mode with more than two simulated arms, you will need backend, mapper, and client changes in addition to the mode class itself.

## Publishing spheres

If your mode owns dynamic spheres, keep the physics object inside the simulator and expose it through `sphere_entities()`.

`CathyThrowSimulation` is the current example. It creates Elastica spheres in `__post_init__` and converts them into `SphereEntity` values each tick:

```python
from virtual_field.core.state import SphereEntity

def sphere_entities(self) -> list[SphereEntity]:
    spheres: list[SphereEntity] = []
    for idx, sphere in enumerate(self.spheres):
        position = np.asarray(sphere.position_collection[..., 0], dtype=np.float64)
        spheres.append(
            SphereEntity(
                sphere_id=f"{self.user_id}_my_mode_sphere_{idx}",
                owner_id=self.user_id,
                translation=position.tolist(),
                radius=float(sphere.radius),
                color_rgb=[0.95, 0.62, 0.32],
            )
        )
    return spheres
```

The backend automatically picks these up:

- once during `register_user()`
- again after every `simulation.step()`

Use stable ids such as `f"{self.user_id}_my_mode_sphere_{idx}"` so updates replace the existing sphere instead of looking like new objects each frame.

## Publishing meshes

For static or generated scenery, override `mesh_entities()` and return `MeshEntity` values. `NoelC4Simulation` uses this for obstacle cylinders.

This is the right place for:

- generated GLTF data URIs
- environment props tied to one user session
- procedurally placed obstacles that the client should render
