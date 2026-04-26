#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

#* uv
.PHONY: uv-download
uv-download:
	curl -LsSf https://astral.sh/uv/install.sh | sh

#* Installation
.PHONY: install
install:
	uv sync --all-groups

.PHONY: pre-commit-install
pre-commit-install:
	uv run --no-sync pre-commit install

#* Formatters
.PHONY: codestyle
codestyle:
	uv run --no-sync black --config pyproject.toml ./

.PHONY: formatting
formatting: codestyle

#* Linting
.PHONY: test
test:
	uv run --no-sync pytest -c pyproject.toml --cov=src --cov-branch

.PHONY: coverage
coverage:
	uv run --no-sync pytest -c pyproject.toml --cov=src --cov-branch --cov-report=term-missing

.PHONY: test_ci
test_ci:
	uv run --no-sync pytest -c pyproject.toml --cov=src --cov-branch --cov-report=xml

.PHONY: check-codestyle
check-codestyle:
	uv run --no-sync black --diff --check --config pyproject.toml ./

.PHONY: mypy
mypy:
	uv run --no-sync mypy --config-file pyproject.toml src

.PHONY: lint
lint: test check-codestyle mypy check-safety

.PHONY: update-dev-deps
update-dev-deps:
	uv add --group dev --upgrade mypy pre-commit pytest coverage pytest-html pytest-cov black

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: mypycache-remove
mypycache-remove:
	find . | grep -E ".mypy_cache" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: exampleresult-remove
exampleresult-remove:
	rm -f examples/*.png
	rm -f examples/*.blend
	rm -f examples/*.blend1

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove mypycache-remove ipynbcheckpoints-remove pytestcache-remove exampleresult-remove
