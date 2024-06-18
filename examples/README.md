# Examples

This directory contains number of examples of how to use this library.

## Installing Requirements

All necessary requirement can be installed as follows:

```bash
poetry instal -n --with examples
```

## Case Examples

Some examples provide additional files or links to published paper for a complete description.
Examples can serve as a starting template for customized usages.

* [ProjectileMotion](./projectile_motion.py)
    * __Purpose__: Illustration case with particle motion under gravity/spring/damping
    * __Features__: Sphere
* [Pendulum2D](./pendelum.py)
    * __Purpose__: Illustration case for collection of primitive geometry
    * __Features__: PrimitiveCollection, Sphere, Cylinder
* [RigidRodSpringMotion](./single_rigid_rod_spring_action.py)
    * __Purpose__: Illustration case for Rod module
    * __Features__: Rod

## Case with External Simulator

* [TimoshenkoBeamCase](./elastica-timoshenko.py)
    * __Purpose__: Example of `elastica_blender` module on simple Timoshenko beam.
    * __Features__: elastica_blender.BlenderRodCallback
