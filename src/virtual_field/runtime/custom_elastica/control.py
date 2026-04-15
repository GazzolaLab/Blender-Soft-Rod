"""Proportional pose tracking as PyElastica external forces and torques.

Used by the Spirob-style runtime to pull one rod element toward a VR-provided
target frame. API reference: :doc:`virtual_field/api/runtime_common_utilities`.
Dual-arm wiring example (targets, attachment, ``TargetPoseProportionalControl``):
:ref:`vf-simple-control-example`.
"""

from typing import Callable

import numpy as np
from elastica.external_forces import NoForces
from numba import njit


class TargetPoseProportionalControl(NoForces):
    r"""PD-style tracking of position (midpoint) and orientation (one segment).

    When ``is_attached()`` is true, applies external forces at the two nodes
    :math:`i` and :math:`i+1` that bound the controlled element, and external
    torque in local director coordinates at node :math:`i`.

    The linear part uses the segment midpoint
    :math:`\mathbf{x}_{\mathrm{mid}} = \tfrac{1}{2}(\mathbf{x}_i + \mathbf{x}_{i+1})`
    and position error :math:`\mathbf{e} = \mathbf{x}_{\mathrm{tar}} - \mathbf{x}_{\mathrm{mid}}`.
    With ramp factor :math:`\alpha(t) = \min\bigl(1,\, t / t_{\mathrm{ramp}}\bigr)`,
    forces are split equally:

    .. math::

        \mathbf{f}_i = \mathbf{f}_{i+1}
        = \tfrac{1}{2}\, k_{\mathrm{lin}}\,\alpha\,\mathbf{e}.

    Directors are stored as row-wise :math:`\mathbf{D}\in\mathbb{R}^{3\times 3}` mapping
    world to local (same convention as PyElastica). Let :math:`\mathbf{C}=\mathbf{D}^{\mathsf{T}}`
    be the corresponding rotation (columns are local axes in world). Orientation error is

    .. math::

        \mathbf{R}_{\mathrm{err}} = \mathbf{C}_{\mathrm{cur}}^{\mathsf{T}}\mathbf{C}_{\mathrm{tar}}
        = \mathbf{D}_{\mathrm{cur}}\mathbf{D}_{\mathrm{tar}}^{\mathsf{T}}.

    The rotation angle :math:`\theta` and axis :math:`\hat{\mathbf{n}}` are recovered from
    :math:`\mathbf{R}_{\mathrm{err}}` (axis--angle). The applied torque is

    .. math::

        \boldsymbol{\tau} = k_{\mathrm{ang}}\,\alpha\,\theta\,\hat{\mathbf{n}}

    in local coordinates, matching ``external_torques``.

    See :ref:`vf-simple-control-example` for a concrete dual-arm usage example
    (``get_target_left`` / ``get_target_right``).

    Parameters
    ----------
    elem_index : int
        Element index along the rod, or negative index counted from the end.
    p_linear_value : float
        :math:`k_{\mathrm{lin}}`, position gain.
    p_angular_value : float
        :math:`k_{\mathrm{ang}}`, angular gain.
    target : Callable[[], tuple[np.ndarray, np.ndarray]]
        Callable returning ``(position, orientation)`` with ``position`` shape
        ``(3,)`` and ``orientation`` shape ``(3, 3)`` (row-director matrix).
    is_attached : Callable[[], bool]
        When false, no forces or torques are applied.
    ramp_up_time : float
        :math:`t_{\mathrm{ramp}}` for :math:`\alpha(t)` (seconds).
    """

    def __init__(
        self,
        elem_index: int,
        p_linear_value: float,
        p_angular_value: float,
        target: Callable[[], tuple[np.ndarray, np.ndarray]],
        is_attached: Callable[[], bool],
        ramp_up_time: float = 1.0,
    ) -> None:
        super().__init__()
        self.elem_index = elem_index
        self.linear_gain = p_linear_value
        self.angular_gain = p_angular_value
        self.ramp_up_time = ramp_up_time
        self.target = target
        self.is_attached = is_attached

    def apply_forces(self, system: object, time: float = 0.0) -> None:
        r"""Gather targets and apply proportional forces/torques (see class Notes).

        Uses simulation time :math:`t` for :math:`\alpha(t)`.
        """
        if not self.is_attached():
            return
        target_position, target_orientation = self.target()

        positions = system.position_collection
        orientations = system.director_collection
        external_forces = system.external_forces
        external_torques = system.external_torques

        elem_count = int(orientations.shape[-1])
        idx = (
            self.elem_index
            if self.elem_index >= 0
            else elem_count + self.elem_index
        )
        idx = max(0, min(elem_count - 1, idx))

        factor = min(1.0, time / max(self.ramp_up_time, 1e-8))
        self.compute(
            external_forces,
            external_torques,
            positions,
            orientations,
            self.linear_gain,
            self.angular_gain,
            factor,
            target_position,
            target_orientation,
            idx,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute(
        external_forces: np.ndarray,
        external_torques: np.ndarray,
        positions: np.ndarray,
        orientations: np.ndarray,
        linear_gain: float,
        angular_gain: float,
        factor: float,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        idx: int,
    ) -> None:
        """Numba kernel implementing the midpoint force and axis-angle torque update."""
        position = 0.5 * (positions[:, idx] + positions[:, idx + 1])
        force = target_position - position
        external_forces[:, idx] += 0.5 * linear_gain * factor * force
        external_forces[:, idx + 1] += 0.5 * linear_gain * factor * force

        orientation = orientations[:, :, idx]
        # row-wise directors: D maps world -> local.
        # Error rotation in current local coordinates:
        # R_err = C_current^T C_target = D_current D_target^T
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

        sin_angle = np.sin(angle)
        if np.abs(sin_angle) >= 1e-8:
            vector = (1.0 / (2.0 * sin_angle)) * np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ]
            )
        else:
            # Near 180 degrees, the skew part is ill-conditioned.
            # Recover a stable axis from the diagonal terms.
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
            vector = np.array([x, y, z])
            norm = np.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
            if norm < 1e-8:
                return
            vector /= norm

        torque = factor * angular_gain * angle * vector
        # external_torques is in local (director) coordinates.
        external_torques[:, idx] += torque
