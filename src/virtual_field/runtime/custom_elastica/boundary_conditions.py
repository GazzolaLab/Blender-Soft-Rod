"""Custom PyElastica forcing utilities for virtual-field simulations.

This module provides growth-aware boundary control for :class:`GrowingCR` rods:
a PD "turret" at the base of the *active* suffix plus discrete extend/retract
triggers that change ``current_elements``.
"""

from typing import Callable

import time as tm

import numpy as np
from elastica.external_forces import NoForces
from numba import njit


@njit(cache=True)  # type: ignore
def _reset_element_kinematics_and_strains(
    positions: np.ndarray,
    directors: np.ndarray,
    element_length: float,
    velocity: np.ndarray,
    omega: np.ndarray,
    sigma: np.ndarray,
    kappa: np.ndarray,
    index: int,
) -> None:
    """Re-seed kinematics after extending the active arm by one element.

    Places node ``index`` one rest length behind node ``index + 1`` along the
    tangent ``directors[2, :, index + 1]``, copies directors from the adjacent
    element, and zeros linear/angular velocity at ``index`` so the new base does
    not inherit stale motion from storage. Strain arrays at ``index`` are left
    to the solver (see commented zero lines in-source).
    """
    positions[:, index] = (
        positions[:, index + 1] - element_length * directors[2, :, index + 1]
    )
    directors[:, :, index] = directors[:, :, index + 1]
    velocity[:, index] = 0.0
    omega[:, index] = 0.0
    # sigma[:, index] = 0.0
    # kappa[:, index] = 0.0


class _GrowingCRBoundaryConditions(NoForces):
    """PD turret + discrete growth for a :class:`GrowingCR` rod (active suffix base).

    PyElastica applies this object as **forcing**. Each substep:

    1. Optionally changes ``system.current_elements`` when a debounced trigger
       fires: **extend** runs :func:`_reset_element_kinematics_and_strains` at
       the new base index; **shrink** only decrements the count (no kinematic reset).
    2. Reads ``controller()`` for a world-frame target **orientation** (3×3).
       ``target_position`` is fixed at construction and is **not** taken from
       the controller return value.
    3. Applies proportional **force** at the base node (index
       ``total_elements - current_elements``) toward ``target_position`` and
       **torque** on the base element to align directors with that orientation.

    The base index moves when ``current_elements`` changes; it is always the
    first node of the simulated (active) suffix, not the physical tip of the rod.
    """

    def __init__(
        self,
        target_position: np.ndarray,
        p_linear_value: float,
        p_angular_value: float,
        controller: Callable[[], tuple[np.ndarray, np.ndarray]],
        trigger_increase_elements: Callable[[], bool],
        trigger_decrease_elements: Callable[[], bool],
        ramp_up_time: float = 1.0,  # Not relevant
    ) -> None:
        """Construct PD turret forcing with optional extend/retract triggers.

        Parameters
        ----------
        target_position
            World-frame point (shape ``(3,)``) the base node is pulled toward.
        p_linear_value, p_angular_value
            Gains on base position error and orientation error (torque axis from
            SO(3) log), scaled by ``factor`` from ``ramp_up_time``.
        controller
            ``() -> (position, orientation)`` with ``orientation`` a 3×3
            rotation matrix. Only ``orientation`` is used; position is ignored.
        trigger_increase_elements, trigger_decrease_elements
            Callables returning ``True`` for one evaluation when the user should
            grow or shrink the active element count (subject to
            ``min_elements``/``total_elements`` and debounce).
        ramp_up_time
            Simulation-time scale for ramping gains from zero to full strength.
        """
        super().__init__()
        if target_position.shape != (3,):
            raise ValueError("target_position must have shape (3,)")
        self.target_position = target_position
        self.linear_gain = p_linear_value
        self.angular_gain = p_angular_value
        self.ramp_up_time = ramp_up_time
        self.controller = controller
        self.trigger_increase_elements = trigger_increase_elements
        self.trigger_decrease_elements = trigger_decrease_elements
        self.last_triggered = -1

    def apply_forces(self, system: object, time: float = 0.0) -> None:
        """Apply growth logic, then PD turret forcing at the active base.

        Parameters
        ----------
        system
            A :class:`GrowingCR` instance (must expose ``total_elements``,
            ``current_elements``, ``min_elements``, PyElastica collections).
        time
            Simulation time used for the linear ramp factor.
        """
        _, ctrl_R = self.controller()
        target_orientation = np.asarray(ctrl_R, dtype=np.float64).reshape(3, 3)

        te = int(system.total_elements)
        ce = int(system.current_elements)
        mn = int(system.min_elements)
        index = te - ce

        # Triggering key
        if tm.time() - self.last_triggered > 0.3:
            if self.trigger_decrease_elements():
                if ce > mn:
                    ce -= 1
                    system.set_current_elements(ce)
                    index = te - ce
                self.last_triggered = tm.time()
            elif self.trigger_increase_elements():
                if ce < te:
                    ce += 1
                    index = te - ce
                    # Need to push-up the element
                    _reset_element_kinematics_and_strains(
                        system.position_collection,
                        system.director_collection,
                        system.rest_lengths[index],
                        system.velocity_collection,
                        system.omega_collection,
                        system.sigma,
                        system.kappa,
                        index,
                    )
                    system.set_current_elements(ce)
                self.last_triggered = tm.time()

        # Turret control
        factor = min(1.0, time / max(self.ramp_up_time, 1e-8))
        self.compute(
            system.external_forces,
            system.external_torques,
            system.position_collection,
            system.director_collection,
            index,
            self.linear_gain,
            self.angular_gain,
            factor,
            self.target_position,
            target_orientation,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute(
        external_forces: np.ndarray,
        external_torques: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        index: int,
        linear_gain: float,
        angular_gain: float,
        factor: float,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
    ) -> None:
        """Add proportional restoring force and torque at the base index.

        Linear part: ``external_forces[:, index] += linear_gain * factor * (target_position - position)``.

        Angular part: build relative rotation ``orientation @ target_orientation.T``,
        map it to a rotation vector (axis–angle / log map) with a stable branch
        near π, then add ``factor * angular_gain * rotation_vector`` to
        ``external_torques[:, index]``.
        """
        position = positions[:, index]
        force = target_position - position
        external_forces[:, index] += linear_gain * factor * force

        # Torque in local coordinates via the SO(3) logarithm / inverse Rodrigues map.
        orientation = orientations[:, :, index]
        rotation = orientation @ target_orientation.T
        trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]
        cos_angle = (trace - 1.0) * 0.5
        if cos_angle < -1.0:
            cos_angle = -1.0
        elif cos_angle > 1.0:
            cos_angle = 1.0
        angle = np.arccos(cos_angle)
        if angle < 1e-8:
            return

        skew_vector = np.array(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )

        if np.abs(np.sin(angle)) >= 1e-8 and np.abs(np.pi - angle) >= 1e-6:
            rotation_vector = 0.5 * angle / np.sin(angle) * skew_vector
        else:
            # Near pi, the skew part becomes ill-conditioned, so recover the axis
            # from the diagonal terms before scaling back to the rotation vector.
            x = np.sqrt(max(0.0, (rotation[0, 0] + 1.0) * 0.5))
            y = np.sqrt(max(0.0, (rotation[1, 1] + 1.0) * 0.5))
            z = np.sqrt(max(0.0, (rotation[2, 2] + 1.0) * 0.5))
            if x >= y and x >= z and x > 1e-8:
                y = (rotation[0, 1] + rotation[1, 0]) / (4.0 * x)
                z = (rotation[0, 2] + rotation[2, 0]) / (4.0 * x)
            elif y >= z and y > 1e-8:
                x = (rotation[0, 1] + rotation[1, 0]) / (4.0 * y)
                z = (rotation[1, 2] + rotation[2, 1]) / (4.0 * y)
            elif z > 1e-8:
                x = (rotation[0, 2] + rotation[2, 0]) / (4.0 * z)
                y = (rotation[1, 2] + rotation[2, 1]) / (4.0 * z)
            axis = np.array([x, y, z])
            norm = np.sqrt(axis[0] ** 2 + axis[1] ** 2 + axis[2] ** 2)
            if norm < 1e-8:
                return
            rotation_vector = angle * axis / norm

        torque = factor * angular_gain * rotation_vector
        external_torques[:, index] += torque
