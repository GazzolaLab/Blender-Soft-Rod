from elastica import NoForces

from numba import njit
import numpy as np
from elastica.typing import RodType


class _SpirobBendConstraint(NoForces):
    def __init__(
        self,
        kt: float,
        allowed_angle_in_deg: float = 30,
    ) -> None:
        super().__init__()
        self.kt = kt
        self.allowed_angle = np.deg2rad(allowed_angle_in_deg)

    def apply_forces(
        self, system: "RodType", time: np.float64 = np.float64(0.0)
    ) -> None:
        del time
        self.compute(
            self.kt,
            self.allowed_angle,
            system.director_collection,
            system.external_torques,
        )

    @staticmethod
    @njit(cache=True)  # type: ignore
    def compute(kt, allowed_angle, director, torques) -> None:
        """
        director: (3, 3, N)
            director[2] is tangent.
        torques: (3, N)
        """
        n_elem = director.shape[2]
        eps = 1.0e-14
        cos_allowed = np.cos(allowed_angle)

        # Enforce maximum relative bend between neighboring elements.
        for i in range(1, n_elem):
            px = director[2, 0, i - 1]
            py = director[2, 1, i - 1]
            pz = director[2, 2, i - 1]

            cx = director[2, 0, i]
            cy = director[2, 1, i]
            cz = director[2, 2, i]

            dot = px * cx + py * cy + pz * cz
            if dot > 1.0:
                dot = 1.0
            elif dot < -1.0:
                dot = -1.0

            # Fast reject: angle <= allowed_angle.
            if dot >= cos_allowed:
                continue

            angle = np.arccos(dot)
            error = angle - allowed_angle
            if error <= 0.0:
                continue

            # Axis rotating previous tangent toward current tangent.
            ax = py * cz - pz * cy
            ay = pz * cx - px * cz
            az = px * cy - py * cx
            axis_norm = np.sqrt(ax * ax + ay * ay + az * az)
            if axis_norm <= eps:
                continue
            ax /= axis_norm
            ay /= axis_norm
            az /= axis_norm

            tau = kt * error
            tau_prev_x = tau * ax
            tau_prev_y = tau * ay
            tau_prev_z = tau * az
            tau_curr_x = -tau_prev_x
            tau_curr_y = -tau_prev_y
            tau_curr_z = -tau_prev_z

            # Convert lab-frame torques to each element's material frame.
            torques[0, i - 1] += (
                director[0, 0, i - 1] * tau_prev_x
                + director[0, 1, i - 1] * tau_prev_y
                + director[0, 2, i - 1] * tau_prev_z
            )
            torques[1, i - 1] += (
                director[1, 0, i - 1] * tau_prev_x
                + director[1, 1, i - 1] * tau_prev_y
                + director[1, 2, i - 1] * tau_prev_z
            )
            torques[2, i - 1] += (
                director[2, 0, i - 1] * tau_prev_x
                + director[2, 1, i - 1] * tau_prev_y
                + director[2, 2, i - 1] * tau_prev_z
            )

            torques[0, i] += (
                director[0, 0, i] * tau_curr_x
                + director[0, 1, i] * tau_curr_y
                + director[0, 2, i] * tau_curr_z
            )
            torques[1, i] += (
                director[1, 0, i] * tau_curr_x
                + director[1, 1, i] * tau_curr_y
                + director[1, 2, i] * tau_curr_z
            )
            torques[2, i] += (
                director[2, 0, i] * tau_curr_x
                + director[2, 1, i] * tau_curr_y
                + director[2, 2, i] * tau_curr_z
            )
