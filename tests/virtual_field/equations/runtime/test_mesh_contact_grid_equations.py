"""Tests for mesh grid lookup and Numba contact fast paths."""

from __future__ import annotations

import numpy as np
import pytest

from virtual_field.runtime.custom_elastica.mesh.mesh_contact_utils import Grid
from virtual_field.runtime.custom_elastica.mesh.mesh_surface import MeshSurface
from virtual_field.runtime.custom_elastica.mesh.rod_mesh_surface_contact import (
    RodMeshSurfaceContactGridMethod,
    RodMeshSurfaceContactGridMethodWithAnisotropicFriction,
)

pytestmark = pytest.mark.equations


def test_find_faces_2d_stacks_position_and_face_indices() -> None:
    """Regression: chunked concat must preserve (element_idx, face_idx) pairing."""
    position_collection = np.array(
        [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    surface_grid = {(0, 0): np.array([0, 1], dtype=np.int64)}
    position_idx, face_idx, element_position = Grid._find_faces_from_2D_grid(
        surface_grid,
        0.0,
        0.0,
        [0, 1],
        10.0,
        position_collection,
        False,
    )
    np.testing.assert_array_equal(position_idx, [0, 0, 1, 1])
    np.testing.assert_array_equal(face_idx, [0, 1, 0, 1])
    assert element_position.shape == (3, 2)


def test_find_faces_3d_stacks_position_and_face_indices() -> None:
    position_collection = np.array(
        [[0.0, 1.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    surface_grid = {(0, 0, 0): np.array([2], dtype=np.int64)}
    position_idx, face_idx, element_position = Grid._find_faces_from_3D_grid(
        surface_grid,
        0.0,
        0.0,
        0.0,
        10.0,
        position_collection,
        False,
    )
    np.testing.assert_array_equal(position_idx, [0, 1])
    np.testing.assert_array_equal(face_idx, [2, 2])
    assert element_position.shape == (3, 2)


def test_find_faces_3d_empty_when_no_grid_cells() -> None:
    position_collection = np.array(
        [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float64
    )
    position_idx, face_idx, element_position = Grid._find_faces_from_3D_grid(
        {},
        0.0,
        0.0,
        0.0,
        10.0,
        position_collection,
        False,
    )
    assert position_idx.size == 0 and face_idx.size == 0
    assert element_position.shape == (3, 1)


def test_rod_mesh_contact_empty_candidates_returns_empty_arrays() -> None:
    n_elem = 1
    n_nodes = 2
    external_forces = np.zeros((3, n_nodes), dtype=np.float64)
    mass = np.ones(n_nodes, dtype=np.float64)
    velocity = np.zeros((3, n_nodes), dtype=np.float64)
    element_position = np.zeros((3, n_elem), dtype=np.float64)
    faces = np.zeros((3, 3, 1), dtype=np.float64)
    face_normals = np.zeros((3, 1), dtype=np.float64)
    face_centers = np.zeros((3, 1), dtype=np.float64)
    side_vectors = np.zeros((3, 3, 1), dtype=np.float64)
    directors = np.zeros((3, 3, n_elem), dtype=np.float64)

    out = RodMeshSurfaceContactGridMethod.rod_mesh_contact(
        faces,
        face_normals,
        face_centers,
        element_position,
        directors,
        side_vectors,
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.float64(1e-4),
        np.float64(1e4),
        np.float64(10.0),
        np.array([0.01], dtype=np.float64),
        mass,
        velocity,
        external_forces,
    )
    plane_mag, no_pen, no_int, normals, ep, ev = out
    assert plane_mag.size == 0
    assert no_pen.size == 0 and no_int.size == 0
    assert normals.shape == (3, 0) and ep.shape == (3, 0) and ev.shape == (3, 0)


def test_mesh_anisotropic_friction_empty_candidates_returns_early() -> None:
    n_elem = 1
    n_nodes = 2
    ext_f = np.zeros((3, n_nodes), dtype=np.float64)
    ext_t = np.zeros((3, n_nodes), dtype=np.float64)
    omega = np.zeros((3, n_elem), dtype=np.float64)
    tangents = np.zeros((3, n_elem), dtype=np.float64)
    directors = np.zeros((3, 3, n_elem), dtype=np.float64)
    radius = np.array([0.01], dtype=np.float64)

    out = RodMeshSurfaceContactGridMethodWithAnisotropicFriction.mesh_anisotropic_friction(
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty((3, 0), dtype=np.float64),
        np.empty(0, dtype=np.int64),
        np.empty((3, 0), dtype=np.float64),
        np.float64(1e-6),
        np.float64(0.1),
        np.float64(0.1),
        np.float64(0.1),
        np.float64(1.0),
        np.float64(0.1),
        np.float64(0.1),
        np.float64(0.1),
        radius,
        tangents,
        directors,
        omega,
        ext_f,
        ext_t,
    )
    no_pen, no_int = out
    assert no_pen.size == 0 and no_int.size == 0


@pytest.fixture
def thin_box_surface() -> MeshSurface:
    """3D grid needs nonzero extent on all axes; a flat triangle yields zero Z cells."""
    pyvista = pytest.importorskip("pyvista")

    mesh = pyvista.Box(bounds=[0, 1, 0, 1, 0, 0.2]).triangulate()
    return MeshSurface(mesh)


def test_grid_find_faces_end_to_end_non_empty(
    thin_box_surface: MeshSurface,
) -> None:
    class _Rod:
        rest_lengths = np.array([0.5], dtype=np.float64)
        radius = np.array([0.05], dtype=np.float64)

    grid = Grid(
        _Rod(),
        thin_box_surface,
        grid_dimension=3,
        exit_boundary_condition=False,
    )
    assert len(grid.surface_grid) > 0

    position_collection = np.array(
        [[0.5, 0.5], [0.5, 0.5], [0.1, 0.1]], dtype=np.float64
    )
    position_idx, face_idx, element_position = grid.find_faces(
        position_collection=position_collection
    )
    assert position_idx.shape == face_idx.shape
    assert element_position.shape[1] == position_collection.shape[1] - 1
    assert position_idx.size > 0
