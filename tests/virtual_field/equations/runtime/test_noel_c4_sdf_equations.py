import numpy as np
import pytest

from virtual_field.runtime.noel_c4_simulation import load_noel_c4_obstacles
from virtual_field.runtime.spirob_elastica.sdf_objects import (
    SDFObstacleCylinders,
    capped_cylinder_sdf,
    capped_cylinder_sdf_and_normal,
    obstacle_cylinder_sdf_and_normal,
)

pytestmark = pytest.mark.equations


def test_load_noel_c4_obstacles() -> None:
    obstacles = load_noel_c4_obstacles()
    assert obstacles.starts.shape == (12, 3)
    assert obstacles.directions.shape == (12, 3)
    assert obstacles.normals.shape == (12, 3)
    assert obstacles.lengths.shape == (12,)
    assert obstacles.radii.shape == (12,)
    assert np.allclose(np.linalg.norm(obstacles.directions, axis=1), 1.0)


def test_capped_cylinder_sdf_regions() -> None:
    start = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 1.0, 0.0])
    normal = np.array([1.0, 0.0, 0.0])

    outside_side = capped_cylinder_sdf(
        np.array([0.3, 0.5, 0.0]), start, direction, 1.0, 0.2
    )
    on_surface = capped_cylinder_sdf(
        np.array([0.2, 0.5, 0.0]), start, direction, 1.0, 0.2
    )
    inside_side, inside_normal = capped_cylinder_sdf_and_normal(
        np.array([0.1, 0.5, 0.0]), start, direction, 1.0, 0.2, normal
    )
    below_cap, below_normal = capped_cylinder_sdf_and_normal(
        np.array([0.0, -0.1, 0.0]), start, direction, 1.0, 0.2, normal
    )
    above_cap = capped_cylinder_sdf(
        np.array([0.0, 1.1, 0.0]), start, direction, 1.0, 0.2
    )

    assert outside_side > 0.0
    assert abs(on_surface) < 1.0e-9
    assert inside_side < 0.0
    assert np.allclose(inside_normal, [1.0, 0.0, 0.0])
    assert below_cap > 0.0
    assert np.allclose(below_normal, [0.0, -1.0, 0.0])
    assert above_cap > 0.0


def test_obstacle_cylinder_contact_force_and_closest_selection() -> None:
    starts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    directions = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    normals = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    lengths = np.array([1.0, 1.0])
    radii = np.array([0.2, 0.2])

    sdf, normal, obstacle_index = obstacle_cylinder_sdf_and_normal(
        np.array([0.15, 0.5, 0.0]), starts, directions, lengths, radii, normals
    )
    assert obstacle_index == 0
    assert sdf < 0.0
    assert np.allclose(normal, [1.0, 0.0, 0.0])

    class DummyRod:
        def __init__(self) -> None:
            self.position_collection = np.array([[0.15], [0.5], [0.0]])
            self.velocity_collection = np.zeros((3, 1))
            self.external_forces = np.zeros((3, 1))
            self.radius = np.array([0.05])

    rod = DummyRod()
    force = SDFObstacleCylinders(
        starts=starts,
        directions=directions,
        lengths=lengths,
        radii=radii,
        normals=normals,
        stiffness=1000.0,
        damping=0.0,
    )
    force.apply_forces(rod)
    assert rod.external_forces[0, 0] > 0.0
    assert np.allclose(rod.external_forces[1:, 0], [0.0, 0.0])

    rod.position_collection = np.array([[1.0], [0.5], [0.0]])
    rod.external_forces[:] = 0.0
    force.apply_forces(rod)
    assert np.allclose(rod.external_forces, 0.0)
