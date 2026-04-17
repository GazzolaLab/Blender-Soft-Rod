import numpy as np
import pytest
from utils import get_mesh_limit

from bsr.geometry.primitives.simple import Cylinder, Sphere

# Visual tolerance for the mesh limit
_VISUAL_ATOL = 1e-7
_VISUAL_RTOL = 1e-4


@pytest.mark.parametrize(
    "center",
    [
        np.array([10, 10, 10]),
        np.array([10, 11, 10]),
        np.array([10, 11, 11]),
        np.array([11, 11, 11]),
    ],
)
@pytest.mark.parametrize("radius", [1, 2, 3, 5.5])
def test_sphere_radius_and_position(center, radius):
    x_min, x_max = center[0] - radius, center[0] + radius
    y_min, y_max = center[1] - radius, center[1] + radius
    z_min, z_max = center[2] - radius, center[2] + radius

    sphere = Sphere(position=center, radius=radius)

    mesh_limit = get_mesh_limit(sphere)

    np.testing.assert_allclose(
        (x_min, x_max, y_min, y_max, z_min, z_max),
        mesh_limit,
        rtol=_VISUAL_RTOL,
        atol=_VISUAL_ATOL,
    )


@pytest.mark.parametrize(
    "position_one",
    [
        np.array([10, 10, 10]),
        np.array([10, 11, -10]),
        np.array([-10, 11, 11]),
    ],
)
@pytest.mark.parametrize("length", [1, 10.5, -1, -10.5])
@pytest.mark.parametrize("radius", [1, 3, 5.5])
def test_x_cylinder_radius_and_positions(position_one, length, radius):
    position_two = position_one + np.array([length, 0, 0])
    y, z = position_one[1], position_one[2]

    x_min, x_max = min(position_one[0], position_two[0]), max(
        position_one[0], position_two[0]
    )
    y_min, y_max = y - radius, y + radius
    z_min, z_max = z - radius, z + radius

    cylinder = Cylinder(
        position_1=position_one, position_2=position_two, radius=radius
    )

    mesh_limit = get_mesh_limit(cylinder)

    np.testing.assert_allclose(
        (x_min, x_max, y_min, y_max, z_min, z_max),
        mesh_limit,
        rtol=_VISUAL_RTOL,
        atol=_VISUAL_ATOL,
    )


@pytest.mark.parametrize(
    "position_one",
    [
        np.array([10, 10, 10]),
        np.array([10, 11, -10]),
        np.array([-10, 11, 11]),
    ],
)
@pytest.mark.parametrize("length", [1, 10.5, -1, -10.5])
@pytest.mark.parametrize("radius", [1, 3, 5.5])
def test_y_cylinder_radius_and_positions(position_one, length, radius):
    position_two = position_one + np.array([0, length, 0])
    x, z = position_one[0], position_one[2]

    x_min, x_max = x - radius, x + radius
    y_min, y_max = min(position_one[1], position_two[1]), max(
        position_one[1], position_two[1]
    )
    z_min, z_max = z - radius, z + radius

    cylinder = Cylinder(
        position_1=position_one, position_2=position_two, radius=radius
    )

    mesh_limit = get_mesh_limit(cylinder)

    print("Expected limits:", (x_min, x_max, y_min, y_max, z_min, z_max))
    print("Actual limits:", mesh_limit)

    np.testing.assert_allclose(
        (x_min, x_max, y_min, y_max, z_min, z_max),
        mesh_limit,
        rtol=_VISUAL_RTOL,
        atol=_VISUAL_ATOL,
    )


@pytest.mark.parametrize(
    "position_one",
    [
        np.array([10, 10, 10]),
        np.array([10, 11, -10]),
        np.array([-10, 11, 11]),
    ],
)
@pytest.mark.parametrize("length", [1, 10.5, -1, -10.5])
@pytest.mark.parametrize("radius", [1, 3, 5.5])
def test_z_cylinder_radius_and_positions(position_one, length, radius):
    position_two = position_one + np.array([0, 0, length])
    x, y = position_one[0], position_one[1]

    x_min, x_max = x - radius, x + radius
    y_min, y_max = y - radius, y + radius
    z_min, z_max = min(position_one[2], position_two[2]), max(
        position_one[2], position_two[2]
    )

    cylinder = Cylinder(
        position_1=position_one, position_2=position_two, radius=radius
    )

    mesh_limit = get_mesh_limit(cylinder)

    np.testing.assert_allclose(
        (x_min, x_max, y_min, y_max, z_min, z_max),
        mesh_limit,
        rtol=_VISUAL_RTOL,
        atol=_VISUAL_ATOL,
    )
