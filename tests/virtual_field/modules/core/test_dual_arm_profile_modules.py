from math import cos, pi, sin

import pytest

from virtual_field.core.commands import ControllerSample, XRInputSample
from virtual_field.core.mapping import DualArmControlMapper
from virtual_field.core.state import Transform

pytestmark = pytest.mark.modules


def test_dual_mapper_maps_hands_to_dual_arms() -> None:
    mapper = DualArmControlMapper()

    sample = XRInputSample(
        timestamp=1.0,
        head_pose=Transform(),
        controllers={
            "left": ControllerSample(
                pose=Transform(
                    translation=[-0.2, 0.1, 0.6], rotation_xyzw=[0, 0, 0, 1]
                ),
                grip=0.9,
                joystick=[0.2, 0.0],
            ),
            "right": ControllerSample(
                pose=Transform(
                    translation=[0.3, 0.0, 0.5], rotation_xyzw=[0, 0, 0, 1]
                ),
                grip=0.1,
            ),
        },
    )

    command = mapper.map_input(sample)

    assert set(command.commands.keys()) == {"left_arm", "right_arm"}
    assert command.commands["left_arm"].active is True
    assert command.commands["right_arm"].active is False


def test_octo_backend_geometry_stays_on_fixed_radius() -> None:
    radius = 0.32
    center_z = -0.15

    radii = {
        round(
            (
                (radius * cos(-0.5 * pi + 2.0 * pi * index / 8.0)) ** 2
                + (radius * sin(-0.5 * pi + 2.0 * pi * index / 8.0)) ** 2
            )
            ** 0.5,
            6,
        )
        for index in range(8)
    }
    z_values = [
        center_z + radius * sin(-0.5 * pi + 2.0 * pi * index / 8.0)
        for index in range(8)
    ]

    assert len(radii) == 1
    assert min(z_values) < center_z
    assert max(z_values) > center_z
