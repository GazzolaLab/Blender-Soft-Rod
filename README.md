# Blender-Soft-Arm-Simulation

## How to build documentation

We will provide external link for documentation once the repository is public.
For now, use the following command to build the documentation:
```sh
cd docs
make clean
make html
open build/html/index.html
```

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
