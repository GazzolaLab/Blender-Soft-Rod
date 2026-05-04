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


class _TipJoystickBending(NoForces):
    """Apply a small joystick-driven bending torque over distal active elements.

    The joystick command is interpreted in the rod's local director frame:
    left/right drives bending about local ``d1`` and forward/back drives bending
    about local ``d2``. Torques are distributed over the last few active
    elements so steering stays smooth instead of hinging at the very tip.
    """

    def __init__(
        self,
        *,
        joystick: Callable[[], np.ndarray],
        gain: float = 0.12,
        smoothing: float = 0.2,
        controlled_elements: int = 6,
    ) -> None:
        super().__init__()
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if controlled_elements <= 0:
            raise ValueError("controlled_elements must be > 0")
        self.joystick = joystick
        self.gain = float(gain)
        self.smoothing = float(smoothing)
        self.controlled_elements = int(controlled_elements)
        self.filtered_command = np.zeros(2, dtype=np.float64)

    def apply_forces(self, system: object, time: float = 0.0) -> None:
        joystick = np.asarray(self.joystick(), dtype=np.float64).reshape(2)
        self.filtered_command += self.smoothing * (joystick - self.filtered_command)
        te = int(system.total_elements)
        ce = int(system.current_elements)
        _apply_tip_joystick_bending(
            system.external_torques,
            te,
            ce,
            self.controlled_elements,
            self.gain,
            self.filtered_command,
        )


@njit(cache=True)
def _apply_tip_joystick_bending(
    external_torques: np.ndarray,
    total_elements: int,
    current_elements: int,
    controlled_elements: int,
    gain: float,
    joystick: np.ndarray,
) -> None:
    if current_elements <= 0:
        return

    command_x = joystick[0]
    command_y = joystick[1]
    magnitude_sq = command_x * command_x + command_y * command_y
    if magnitude_sq <= 1.0e-10:
        return

    distal_count = controlled_elements
    if distal_count > current_elements:
        distal_count = current_elements

    start = total_elements - distal_count
    torque_x = -gain * command_y
    torque_y = gain * command_x

    for offset in range(distal_count):
        idx = start + offset
        weight = (offset + 1.0) / distal_count
        external_torques[0, idx] += weight * torque_x
        external_torques[1, idx] += weight * torque_y


class _TravelingContractingWave(NoForces):
    """Send a one-shot contraction and stiffness pulse toward the tip."""

    def __init__(
        self,
        *,
        event_id: Callable[[], int],
        original_rest_sigma: np.ndarray,
        original_shear_matrix: np.ndarray,
        original_bend_matrix: np.ndarray,
        amplitude: float = -0.12,
        stiffness_amplitude: float = 0.25,
        width: float = 4.0,
        duration: float = 0.45,
    ) -> None:
        super().__init__()
        if width <= 0.0:
            raise ValueError("width must be > 0")
        if duration <= 0.0:
            raise ValueError("duration must be > 0")
        self.event_id = event_id
        self.original_rest_sigma = np.array(
            original_rest_sigma, dtype=np.float64, copy=True
        )
        self.original_shear_matrix = np.array(
            original_shear_matrix, dtype=np.float64, copy=True
        )
        self.original_bend_matrix = np.array(
            original_bend_matrix, dtype=np.float64, copy=True
        )
        self.amplitude = float(amplitude)
        self.stiffness_amplitude = float(stiffness_amplitude)
        self.width = float(width)
        self.duration = float(duration)
        self._last_event_id = int(self.event_id())
        self._wave_start_time = -1.0

    def apply_forces(self, system: object, time: float = 0.0) -> None:
        current_event_id = int(self.event_id())
        if current_event_id != self._last_event_id:
            self._last_event_id = current_event_id
            self._wave_start_time = time

        elapsed = -1.0
        if self._wave_start_time >= 0.0:
            elapsed = time - self._wave_start_time
            if elapsed > self.duration:
                self._wave_start_time = -1.0
                elapsed = -1.0

        _apply_traveling_contracting_wave(
            system.rest_sigma,
            system.shear_matrix,
            system.bend_matrix,
            self.original_rest_sigma,
            self.original_shear_matrix,
            self.original_bend_matrix,
            int(system.total_elements),
            int(system.current_elements),
            self.amplitude,
            self.stiffness_amplitude,
            self.width,
            self.duration,
            elapsed,
        )


@njit(cache=True)
def _apply_traveling_contracting_wave(
    rest_sigma: np.ndarray,
    shear_matrix: np.ndarray,
    bend_matrix: np.ndarray,
    original_rest_sigma: np.ndarray,
    original_shear_matrix: np.ndarray,
    original_bend_matrix: np.ndarray,
    total_elements: int,
    current_elements: int,
    amplitude: float,
    stiffness_amplitude: float,
    width: float,
    duration: float,
    elapsed: float,
) -> None:
    rest_sigma[2, :] = original_rest_sigma[2, :]
    shear_matrix[0, 0, :] = original_shear_matrix[0, 0, :]
    shear_matrix[1, 1, :] = original_shear_matrix[1, 1, :]
    bend_matrix[0, 0, :] = original_bend_matrix[0, 0, :]
    bend_matrix[1, 1, :] = original_bend_matrix[1, 1, :]
    if elapsed < 0.0 or current_elements <= 0:
        return

    active_start = total_elements - current_elements
    travel = elapsed / duration
    if travel < 0.0:
        travel = 0.0
    elif travel > 1.0:
        travel = 1.0
    center = active_start + travel * max(current_elements - 1, 0)

    width_sq = width * width
    for idx in range(active_start, total_elements):
        distance = idx - center
        weight = np.exp(-(distance * distance) / (2.0 * width_sq))
        rest_sigma[2, idx] += amplitude * weight
        stiffness_scale = 1.0 + stiffness_amplitude * weight
        shear_matrix[0, 0, idx] = original_shear_matrix[0, 0, idx] * stiffness_scale
        shear_matrix[1, 1, idx] = original_shear_matrix[1, 1, idx] * stiffness_scale
        bend_matrix[0, 0, idx] = original_bend_matrix[0, 0, idx] * stiffness_scale
        bend_matrix[1, 1, idx] = original_bend_matrix[1, 1, idx] * stiffness_scale
