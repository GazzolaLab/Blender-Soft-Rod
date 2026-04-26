from collections.abc import Callable

import numpy as np
from elastica.external_forces import NoForces
from numba import njit


class _SphereBoxed(NoForces):
    def __init__(
        self,
        *,
        bounding_box: np.ndarray,
        stiffness: float = 1e3,
        damping: float = 1.0,
    ) -> None:
        super().__init__()
        self.bounding_box = np.asarray(bounding_box, dtype=np.float64)
        if self.bounding_box.shape != (2, 3):
            raise ValueError("bounding_box must have shape (2, 3)")
        if np.any(self.bounding_box[0] >= self.bounding_box[1]):
            raise ValueError(
                "bounding_box min corner must be strictly below max corner"
            )
        self.stiffness = stiffness
        self.damping = damping
        self.gravity = np.array([0.0, -9.81, 0.0])

    def apply_forces(self, system, time: float = 0.0) -> None:
        sphere = system
        sphere.external_forces[..., 0] += _boxed_sphere_force(
            sphere.position_collection[..., 0],
            sphere.velocity_collection[..., 0],
            sphere.radius,
            sphere.mass[0],
            self.bounding_box,
            self.stiffness,
            self.damping,
            self.gravity,
        )


@njit(cache=True)
def _boxed_sphere_force(
    position: np.ndarray,
    velocity: np.ndarray,
    radius: float,
    mass: float,
    bounding_box: np.ndarray,
    stiffness: float,
    damping: float,
    gravity: np.ndarray,
) -> np.ndarray:
    force = np.zeros((3,), dtype=np.float64)
    for axis in range(3):
        force[axis] += mass * gravity[axis]
    for axis in range(3):
        lower = bounding_box[0, axis] + radius
        upper = bounding_box[1, axis] - radius
        if position[axis] <= lower:
            penetration = lower - position[axis]
            force[axis] += stiffness * penetration - damping * velocity[axis]
        elif position[axis] >= upper:
            penetration = position[axis] - upper
            force[axis] += -stiffness * penetration - damping * velocity[axis]
    return force


class _PullSphereToPoint(NoForces):
    def __init__(
        self,
        *,
        target: Callable[[], np.ndarray],
        is_active: Callable[[], bool],
        stiffness: float = 18.0,
        damping: float = 12.0,
        max_force: float = 40.0,
    ) -> None:
        super().__init__()
        self.target = target
        self.is_active = is_active
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.max_force = float(max_force)

    def apply_forces(self, system: object, time: float = 0.0) -> None:
        if not self.is_active():
            return

        sphere = system
        target = np.asarray(self.target(), dtype=np.float64).reshape(3)
        sphere.external_forces[..., 0] += _pull_sphere_to_point_force(
            sphere.position_collection[..., 0],
            sphere.velocity_collection[..., 0],
            target,
            self.stiffness,
            self.damping,
            self.max_force,
        )


@njit(cache=True)
def _pull_sphere_to_point_force(
    position: np.ndarray,
    velocity: np.ndarray,
    target: np.ndarray,
    stiffness: float,
    damping: float,
    max_force: float,
) -> np.ndarray:
    delta = target - position
    distance = np.sqrt(np.dot(delta, delta))
    if distance <= 1.0e-8:
        return np.zeros(3, dtype=np.float64)

    direction = delta / distance
    speed_along_line = np.dot(velocity, direction)
    magnitude = stiffness * distance - damping * speed_along_line
    if magnitude <= 0.0:
        return np.zeros(3, dtype=np.float64)
    if magnitude > max_force:
        magnitude = max_force
    return magnitude * direction
