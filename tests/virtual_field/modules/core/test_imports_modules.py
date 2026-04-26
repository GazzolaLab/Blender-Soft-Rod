import importlib

import pytest

pytestmark = pytest.mark.modules


def test_bsr_core_import_without_blender_runtime() -> None:
    module = importlib.import_module("virtual_field.core")
    assert hasattr(module, "SceneState")
