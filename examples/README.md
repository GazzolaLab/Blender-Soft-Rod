# Examples

This directory contains small examples that exercise the Blender and
`elastica_blender` APIs.

## Installing requirements

The repository does not define a separate `examples` dependency group.
Install the development environment from the project root instead:

```bash
uv sync --all-groups
```

You can then run an example with `uv run python`, for example:

```bash
uv run python examples/projectile_motion.py
```

## Example scripts

Some examples provide additional files or links to published paper for a complete description.
Examples can serve as a starting template for customized usages.

- [ProjectileMotion](./projectile_motion.py): particle motion under
  gravity/spring/damping using a sphere primitive.
- [Pendulum2D](./pendulum.py): simple primitive-geometry scene using spheres and
  cylinders.
- [PoseDemo](./pose_demo.py): pose composition and placement example.
- [CameraMovement](./camera_movement.py): camera motion and scene framing demo.
- [RigidRodSpringMotion](./single_rigid_rod_spring_action.py): rod animation
  example.

## Example with external simulator

- [TimoshenkoBeamCase](./elastica_timoshenko.py): example of
  `elastica_blender.BlenderRodCallback` on a simple Timoshenko beam.
