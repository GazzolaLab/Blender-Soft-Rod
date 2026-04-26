<div align="center">
<h1> Soft-arm Platform for Action, Rendering, and Control </h1>

<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/>
<a href='https://softarmplatformforactionrendringcontrol.readthedocs.io/en/latest'>
    <img src='https://readthedocs.org/projects/softarmplatformforactionrendringcontrol/badge/?version=latest' alt='Documentation Status' />
</a>
<a href='https://github.com/GazzolaLab/SPARC/actions'>
    <img src='https://github.com/GazzolaLab/SPARC/actions/workflows/main.yml/badge.svg' alt='CI' />
</a>
<a href='https://codecov.io/gh/GazzolaLab/SPARC'>
    <img src='https://codecov.io/gh/GazzolaLab/SPARC/branch/main/graph/badge.svg' alt='Coverage' />
</a>

</div>

----

> Repository is still under development.

The Blender Soft Arm (bsr) includes data visualization tools and analysis for soft-arm robotics data.
The slender body is defined by a series of points and radius, and the data is visualized in [Blender](https://www.blender.org/).

## How to install

Easiest way to install the stable version of the package is to use `pip`:

```sh
pip install bsr
```

## Examples

We provide minimal [example scripts](.examples) to demonstrate the usage of the package.

## Development version

The development version includes unit-tests, documentation, examples, and other development tools.
We primarily use [`uv`](https://docs.astral.sh/uv/) to manage the dependencies and the development environment.
Necessary commands are provided in the `Makefile`.

```sh
git clone https://github.com/GazzolaLab/SPARC.git
cd SPARC
make install  # Assuming you have uv installed.
make pre-commit-install
```

Below are additional commands that you can use to manage the development environment.

- Documentation

```sh
cd docs
make clean
make html
open build/html/index.html
```

- Unittests

```sh
make test
make coverage
```

- Code formatting

```sh
make formatting
```

- Check type-hinting

```sh
make mypy
```
