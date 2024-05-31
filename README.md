# Blender-Soft-Arm-Simulation

## How setup development environment

We are using `poetry` to maintain the dependency trees for this project. To install `poetry` run the following command:

```sh
# https://python-poetry.org/docs/#installing-with-the-official-installer
make poetry-download
```

To remove the poetry, simply run `make poetry-remove`.

To install the dependencies for development, run the following command:

```sh
make install
make pre-commit-install
```

### Unittests

```sh
make test
```

### Code formatting

```sh
make formatting
```

### Check type-hinting

```sh
make mypy
```

# Projectile Motion
  - Projectile Motion: graphs the movement of an object with the projectile motion formula
  - Projectile Motion Final: Plotting projectile motion used discretized methods
  - Proejctile Motion (-ky) as gravity: Replaces gravity with a spring constant to create conservative oscillatory motion
  - Projectile Motion (-ky - bv): Replaces gravity with -ky -bv to create damped oscialltory motion

# Blender
  - rod_sim.py: The python script for a soft arm (spherical joints which are connected by rod linkages) performing oscillatory motion. The lengths of the linkages change as the joints seperate from each other, and the arm maintains its shape through the stretch.
  - Rod_simulation.blender: The blender animation for the arm performing the oscillatory motion.
