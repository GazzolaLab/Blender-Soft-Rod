import numpy as np
import pytest

from virtual_field.runtime.custom_elastica.contacts import (
    SuckerActuationToSphere,
)
from virtual_field.runtime.custom_elastica.forcing import (
    _boxed_sphere_force,
    _pull_sphere_to_point_force,
    _PullSphereToPoint,
    _SphereBoxed,
)

pytestmark = pytest.mark.equations


class DummySphere:
    def __init__(
        self,
        *,
        position: list[float],
        velocity: list[float],
        radius: float,
        mass: float,
    ) -> None:
        self.position_collection = np.asarray(
            position, dtype=np.float64
        ).reshape(3, 1)
        self.velocity_collection = np.asarray(
            velocity, dtype=np.float64
        ).reshape(3, 1)
        self.radius = float(radius)
        self.mass = np.asarray([mass], dtype=np.float64)
        self.external_forces = np.zeros((3, 1), dtype=np.float64)
        self.external_torques = np.zeros((3, 1), dtype=np.float64)
        self.director_collection = np.zeros((3, 3, 1), dtype=np.float64)
        self.director_collection[..., 0] = np.eye(3, dtype=np.float64)
        self.omega_collection = np.zeros((3, 1), dtype=np.float64)


class DummyRod:
    def __init__(self) -> None:
        self.position_collection = np.asarray(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, -0.1],
            ],
            dtype=np.float64,
        )
        self.velocity_collection = np.zeros((3, 2), dtype=np.float64)
        self.radius = np.asarray([0.02], dtype=np.float64)
        self.lengths = np.asarray([0.1], dtype=np.float64)
        self.director_collection = np.zeros((3, 3, 1), dtype=np.float64)
        self.director_collection[0, :, 0] = np.asarray([1.0, 0.0, 0.0])
        self.director_collection[1, :, 0] = np.asarray([0.0, 1.0, 0.0])
        self.director_collection[2, :, 0] = np.asarray([0.0, 0.0, -1.0])
        self.external_forces = np.zeros((3, 2), dtype=np.float64)
        self.external_torques = np.zeros((3, 1), dtype=np.float64)


def test_boxed_sphere_force_adds_gravity_and_wall_response() -> None:
    force = _boxed_sphere_force(
        position=np.asarray([-0.3, 0.1, 0.0], dtype=np.float64),
        velocity=np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
        radius=0.1,
        mass=2.0,
        bounding_box=np.asarray(
            [[-0.2, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64
        ),
        stiffness=100.0,
        damping=10.0,
        gravity=np.asarray([0.0, -9.81, 0.0], dtype=np.float64),
    )
    assert force[0] > 0.0
    assert np.isclose(force[1], -19.62)
    assert np.isclose(force[2], 0.0)


def test_sphere_boxed_apply_forces_updates_dummy_sphere() -> None:
    sphere = DummySphere(
        position=[0.0, -0.2, 0.0],
        velocity=[0.0, 0.0, 0.0],
        radius=0.1,
        mass=1.5,
    )
    forcing = _SphereBoxed(
        bounding_box=np.asarray(
            [[-1.0, 0.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64
        ),
        stiffness=50.0,
        damping=0.0,
    )
    forcing.apply_forces(sphere)
    assert sphere.external_forces[1, 0] > 0.0


def test_boxed_sphere_force_damps_outward_velocity_at_exact_wall_contact() -> (
    None
):
    force = _boxed_sphere_force(
        position=np.asarray([-0.1, 0.0, 0.0], dtype=np.float64),
        velocity=np.asarray([-2.0, 0.0, 0.0], dtype=np.float64),
        radius=0.1,
        mass=1.0,
        bounding_box=np.asarray(
            [[-0.2, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64
        ),
        stiffness=100.0,
        damping=10.0,
        gravity=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
    )
    assert force[0] > 0.0


def test_sphere_boxed_rejects_invalid_bounding_box_order() -> None:
    try:
        _SphereBoxed(
            bounding_box=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float64
            )
        )
    except ValueError as exc:
        assert "bounding_box" in str(exc)
    else:
        raise AssertionError(
            "expected invalid bounding_box ordering to raise ValueError"
        )


def test_pull_sphere_to_point_force_points_to_target_and_clamps() -> None:
    force = _pull_sphere_to_point_force(
        position=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        velocity=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        target=np.asarray([10.0, 0.0, 0.0], dtype=np.float64),
        stiffness=100.0,
        damping=0.0,
        max_force=25.0,
    )
    assert np.allclose(force, [25.0, 0.0, 0.0])


def test_pull_sphere_to_point_respects_activation() -> None:
    sphere = DummySphere(
        position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], radius=0.1, mass=1.0
    )
    forcing = _PullSphereToPoint(
        target=lambda: np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        is_active=lambda: False,
        stiffness=20.0,
        damping=0.0,
        max_force=100.0,
    )
    forcing.apply_forces(sphere)
    assert np.allclose(sphere.external_forces[:, 0], 0.0)


def test_sucker_actuation_only_applies_when_triggered() -> None:
    rod = DummyRod()
    sphere = DummySphere(
        position=[0.03, 0.0, -0.05],
        velocity=[0.0, 0.0, 0.0],
        radius=0.01,
        mass=1.0,
    )
    contact = SuckerActuationToSphere(
        k=10.0, nu=0.0, trigger=False, capture_distance=0.2
    )

    contact.apply_contact(rod, sphere)
    assert np.allclose(sphere.external_forces[:, 0], 0.0)
    assert np.allclose(rod.external_forces, 0.0)
    assert np.allclose(rod.external_torques, 0.0)

    contact.set_trigger(True)
    contact.apply_contact(rod, sphere)
    assert sphere.external_forces[0, 0] < 0.0
    assert np.allclose(
        rod.external_forces.sum(axis=1), -sphere.external_forces[:, 0]
    )
    assert np.allclose(rod.velocity_collection, 0.0)


def test_sucker_actuation_force_decays_with_distance() -> None:
    rod = DummyRod()
    near_sphere = DummySphere(
        position=[0.03, 0.0, -0.05],
        velocity=[0.0, 0.0, 0.0],
        radius=0.01,
        mass=1.0,
    )
    far_sphere = DummySphere(
        position=[0.15, 0.0, -0.05],
        velocity=[0.0, 0.0, 0.0],
        radius=0.01,
        mass=1.0,
    )
    contact = SuckerActuationToSphere(
        k=10.0, nu=0.0, trigger=True, capture_distance=0.2
    )

    contact.apply_contact(rod, near_sphere)
    contact.apply_contact(rod, far_sphere)

    assert np.linalg.norm(near_sphere.external_forces[:, 0]) > np.linalg.norm(
        far_sphere.external_forces[:, 0]
    )


def test_sucker_actuation_adds_alignment_torque() -> None:
    rod = DummyRod()
    sphere = DummySphere(
        position=[0.03, 0.0, -0.02],
        velocity=[0.0, 0.0, 0.0],
        radius=0.01,
        mass=1.0,
    )
    contact = SuckerActuationToSphere(
        k=10.0,
        nu=0.0,
        trigger=True,
        capture_distance=0.2,
        alignment_torque_scale=1.0,
    )

    contact.apply_contact(rod, sphere)

    assert np.linalg.norm(sphere.external_forces[:, 0]) > 0.0
    assert np.linalg.norm(sphere.external_torques[:, 0]) > 0.0
    assert np.linalg.norm(rod.external_torques[:, 0]) > 0.0
