from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from virtual_field.core.state import Transform


@pytest.fixture
def controller_transform() -> Callable[[list[float], list[float]], Transform]:
    def _make(
        translation: list[float], rotation_xyzw: list[float]
    ) -> Transform:
        return Transform(translation=translation, rotation_xyzw=rotation_xyzw)

    return _make


@pytest.fixture
def quat_from_euler_xyz() -> Callable[[float, float, float], list[float]]:
    def _make(roll: float, pitch: float, yaw: float) -> list[float]:
        cr = math.cos(0.5 * roll)
        sr = math.sin(0.5 * roll)
        cp = math.cos(0.5 * pitch)
        sp = math.sin(0.5 * pitch)
        cy = math.cos(0.5 * yaw)
        sy = math.sin(0.5 * yaw)
        return [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]

    return _make
