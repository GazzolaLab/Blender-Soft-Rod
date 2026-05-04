import numpy as np
import pytest

from virtual_field.runtime.spirob_elastica.sdf_objects import (
    SDFObstacleCylinders,
    obstacle_cylinder_sdf_and_normal,
)
from virtual_field.runtime.spirob_elastica.sdf_objects_hash import (
    CylinderSpatialHash,
    SDFObstacleCylindersHash,
    obstacle_cylinder_sdf_and_normal_hash,
)

pytestmark = pytest.mark.equations


def _make_obstacles() -> tuple[np.ndarray, ...]:
    starts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 0.0, 1.5],
        ],
        dtype=np.float64,
    )
    directions = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    lengths = np.array([1.0, 1.0, 0.75], dtype=np.float64)
    radii = np.array([0.2, 0.2, 0.15], dtype=np.float64)
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    return starts, directions, lengths, radii, normals


def test_cylinder_spatial_hash_query_returns_local_candidates() -> None:
    starts, directions, lengths, radii, normals = _make_obstacles()
    spatial_hash = CylinderSpatialHash(
        starts=starts,
        directions=directions,
        lengths=lengths,
        radii=radii,
        normals=normals,
        cell_size=0.75,
    )

    candidates = spatial_hash.query_candidates(np.array([0.1, 0.5, 0.0]))

    assert 0 in candidates
    assert 1 not in candidates
    assert 2 not in candidates


@pytest.mark.parametrize(
    "point",
    [
        np.array([0.1, 0.5, 0.0], dtype=np.float64),
        np.array([1.65, 0.25, 0.0], dtype=np.float64),
        np.array([0.4, 0.1, 1.55], dtype=np.float64),
        np.array([0.9, 0.9, 0.9], dtype=np.float64),
    ],
)
def test_hash_sdf_matches_bruteforce_query(point: np.ndarray) -> None:
    starts, directions, lengths, radii, normals = _make_obstacles()
    spatial_hash = CylinderSpatialHash(
        starts=starts,
        directions=directions,
        lengths=lengths,
        radii=radii,
        normals=normals,
        cell_size=0.75,
    )

    sdf_ref, normal_ref, idx_ref = obstacle_cylinder_sdf_and_normal(
        point, starts, directions, lengths, radii, normals
    )
    sdf_hash, normal_hash, idx_hash = obstacle_cylinder_sdf_and_normal_hash(
        point,
        starts,
        directions,
        lengths,
        radii,
        normals,
        spatial_hash,
    )

    assert idx_hash == idx_ref
    assert np.isclose(sdf_hash, sdf_ref)
    assert np.allclose(normal_hash, normal_ref)


def test_hash_force_matches_bruteforce_force_and_tip_penetration() -> None:
    starts, directions, lengths, radii, normals = _make_obstacles()
    positions = np.array(
        [
            [0.15, 0.15, 1.65],
            [0.25, 0.50, 0.50],
            [0.00, 0.00, 0.00],
        ],
        dtype=np.float64,
    )
    velocities = np.array(
        [
            [0.0, -0.1, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    radius = np.array([0.05, 0.05, 0.05], dtype=np.float64)

    class DummyRod:
        def __init__(self) -> None:
            self.position_collection = positions.copy()
            self.velocity_collection = velocities.copy()
            self.external_forces = np.zeros((3, positions.shape[1]), dtype=np.float64)
            self.radius = radius.copy()

    tip_state_ref = {"arm": 0.0}
    tip_state_hash = {"arm": 0.0}

    rod_ref = DummyRod()
    rod_hash = DummyRod()

    brute_force = SDFObstacleCylinders(
        starts=starts,
        directions=directions,
        lengths=lengths,
        radii=radii,
        normals=normals,
        tip_penetration_state=tip_state_ref,
        tip_penetration_key="arm",
        stiffness=1000.0,
        damping=0.5,
    )
    hash_force = SDFObstacleCylindersHash(
        starts=starts,
        directions=directions,
        lengths=lengths,
        radii=radii,
        normals=normals,
        tip_penetration_state=tip_state_hash,
        tip_penetration_key="arm",
        stiffness=1000.0,
        damping=0.5,
        cell_size=0.75,
    )

    brute_force.apply_forces(rod_ref)
    hash_force.apply_forces(rod_hash)

    assert np.allclose(rod_hash.external_forces, rod_ref.external_forces)
    assert tip_state_hash["arm"] == pytest.approx(tip_state_ref["arm"])


def test_hash_force_auto_expands_query_padding_for_larger_rod_radius() -> None:
    starts, directions, lengths, radii, normals = _make_obstacles()

    class DummyRod:
        def __init__(self) -> None:
            self.position_collection = np.array([[0.15], [0.25], [0.0]])
            self.velocity_collection = np.zeros((3, 1), dtype=np.float64)
            self.external_forces = np.zeros((3, 1), dtype=np.float64)
            self.radius = np.array([0.3], dtype=np.float64)

    rod = DummyRod()
    force = SDFObstacleCylindersHash(
        starts=starts,
        directions=directions,
        lengths=lengths,
        radii=radii,
        normals=normals,
        stiffness=1000.0,
        damping=0.0,
    )

    assert force.query_padding == pytest.approx(np.max(radii))

    force.apply_forces(rod)

    assert force.query_padding == pytest.approx(0.3)
    assert rod.external_forces[0, 0] > 0.0
