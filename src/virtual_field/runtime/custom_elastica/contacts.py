from __future__ import annotations

from collections.abc import Callable
from typing import Type

import numpy as np
from numba import njit

if not hasattr(np, "typing"):
    import numpy.typing as np_typing

    np.typing = np_typing  # type: ignore[attr-defined]

from elastica import NoContact, Sphere
from elastica.typing import RodType


class SuckerActuation(NoContact):
    """Octopus-sucker attraction between a rod and a sphere.

    Each rod element owns a sucker located at
    `element_center + director[0] * element_radius`.
    When the trigger is active, the sphere is attracted toward each sucker with a
    magnitude that decays rapidly with signed distance from the sphere surface.
    """

    def __init__(
        self,
        k: float,
        nu: float,
        *,
        trigger: bool | Callable[[], bool] = False,
        capture_distance: float = 0.05,
        min_distance: float = 1.0e-6,
        alignment_torque_scale: float = 10,
    ) -> None:
        super().__init__()
        self.k = float(k)
        self.nu = float(nu)
        self.capture_distance = float(capture_distance)
        self.min_distance = float(min_distance)
        self.alignment_torque_scale = float(alignment_torque_scale)
        self.trigger = trigger
        self.trigger_active = bool(trigger) if isinstance(trigger, bool) else False

    @property
    def _allowed_system_two(self) -> list[Type]:
        return [Sphere]

    def set_trigger(self, active: bool) -> None:
        self.trigger_active = bool(active)

    def is_triggered(self) -> bool:
        if callable(self.trigger):
            return bool(self.trigger())
        return self.trigger_active

    def apply_contact(
        self,
        system_one: RodType,
        system_two: Sphere,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        if not self.is_triggered():
            return

        if _prune_using_aabbs_rod_sphere_impl(
            system_one.position_collection,
            system_one.radius,
            system_one.lengths,
            system_two.position_collection[..., 0],
            system_two.radius,
            self.capture_distance,
        ):
            return

        sphere_center = system_two.position_collection[..., 0]
        sphere_velocity = system_two.velocity_collection[..., 0]
        sphere_radius = system_two.radius

        rod_positions = system_one.position_collection
        rod_element_velocities = 0.5 * (system_one.velocity_collection[..., 1:] + system_one.velocity_collection[..., :-1])
        rod_directors = system_one.director_collection
        rod_radii = system_one.radius

        _compute_sucker_sphere_force(
            rod_positions,
            rod_element_velocities,
            rod_directors,
            rod_radii,
            sphere_center,
            sphere_velocity,
            sphere_radius,
            self.k,
            self.nu,
            self.capture_distance,
            self.min_distance,
            self.alignment_torque_scale,
            system_two.director_collection,
            system_one.external_forces,
            system_one.external_torques,
            system_two.external_forces,
            system_two.external_torques,
        )


@njit(cache=True)
def _prune_using_aabbs_rod_sphere_impl(
    rod_positions: np.ndarray,
    rod_radii: np.ndarray,
    rod_lengths: np.ndarray,
    sphere_center: np.ndarray,
    sphere_radius: float,
    capture_distance: float,
) -> bool:
    max_radius = 0.0
    for idx in range(rod_radii.shape[0]):
        if rod_radii[idx] > max_radius:
            max_radius = rod_radii[idx]

    half_max_length = 0.0
    for idx in range(rod_lengths.shape[0]):
        candidate = 0.5 * rod_lengths[idx]
        if candidate > half_max_length:
            half_max_length = candidate

    rod_padding = max(max_radius, half_max_length) + capture_distance

    rod_min = np.empty(3, dtype=np.float64)
    rod_max = np.empty(3, dtype=np.float64)
    for axis in range(3):
        axis_min = rod_positions[axis, 0]
        axis_max = rod_positions[axis, 0]
        for node_idx in range(1, rod_positions.shape[1]):
            value = rod_positions[axis, node_idx]
            if value < axis_min:
                axis_min = value
            if value > axis_max:
                axis_max = value
        rod_min[axis] = axis_min - rod_padding
        rod_max[axis] = axis_max + rod_padding

    sphere_padding = sphere_radius + capture_distance
    for axis in range(3):
        sphere_min = sphere_center[axis] - sphere_padding
        sphere_max = sphere_center[axis] + sphere_padding
        if rod_max[axis] < sphere_min or sphere_max < rod_min[axis]:
            return True
    return False


@njit(cache=True)
def _compute_sucker_sphere_force(
    rod_positions: np.ndarray,
    rod_element_velocities: np.ndarray,
    rod_directors: np.ndarray,
    rod_radii: np.ndarray,
    sphere_center: np.ndarray,
    sphere_velocity: np.ndarray,
    sphere_radius: float,
    stiffness: float,
    damping: float,
    capture_distance: float,
    min_distance: float,
    alignment_torque_scale: float,
    sphere_directors: np.ndarray,
    rod_external_forces: np.ndarray,
    rod_external_torques: np.ndarray,
    sphere_external_forces: np.ndarray,
    sphere_external_torques: np.ndarray,
) -> None:
    sphere_force = np.zeros(3, dtype=np.float64)
    sphere_torque_world = np.zeros(3, dtype=np.float64)
    element_count = rod_radii.shape[0]

    for idx in range(element_count):
        element_center = np.empty(3, dtype=np.float64)
        sucker_position = np.empty(3, dtype=np.float64)
        for axis in range(3):
            element_center[axis] = 0.5 * (
                rod_positions[axis, idx + 1] + rod_positions[axis, idx]
            )
            sucker_position[axis] = (
                element_center[axis] + rod_directors[0, axis, idx] * rod_radii[idx]
            )

        center_to_sucker = sucker_position - sphere_center
        center_distance = np.sqrt(np.dot(center_to_sucker, center_to_sucker))
        if center_distance <= min_distance:
            continue

        surface_normal = center_to_sucker / center_distance
        sphere_surface_point = sphere_center + sphere_radius * surface_normal
        attachment_vector = sucker_position - sphere_surface_point
        attachment_distance = np.sqrt(np.dot(attachment_vector, attachment_vector))
        if attachment_distance <= min_distance:
            attachment_direction = surface_normal
        else:
            attachment_direction = attachment_vector / attachment_distance

        signed_distance = center_distance - sphere_radius
        if signed_distance > capture_distance:
            continue

        positive_distance = signed_distance
        if positive_distance < 0.0:
            positive_distance = 0.0

        distance_scale = capture_distance
        if distance_scale < min_distance:
            distance_scale = min_distance

        normalized_distance = positive_distance / distance_scale
        distance_weight = np.exp(-(normalized_distance * normalized_distance) * 9.0)

        relative_velocity = rod_element_velocities[:, idx] - sphere_velocity
        closing_speed = np.dot(relative_velocity, attachment_direction)
        magnitude = stiffness * distance_weight + damping * closing_speed
        if magnitude <= 0.0:
            continue

        pair_force = magnitude * attachment_direction
        sphere_force += pair_force

        reaction_force = -pair_force
        for axis in range(3):
            rod_external_forces[axis, idx] += 0.5 * reaction_force[axis]
            rod_external_forces[axis, idx + 1] += 0.5 * reaction_force[axis]

        center_to_contact = sphere_surface_point - sphere_center
        center_to_element = element_center - sphere_center
        alignment_torque_world = np.zeros(3, dtype=np.float64)
        element_distance = np.sqrt(np.dot(center_to_element, center_to_element))
        if element_distance > min_distance:
            desired_contact_direction = center_to_element / element_distance
            torque_axis = np.empty(3, dtype=np.float64)
            torque_axis[0] = (
                center_to_contact[1] * desired_contact_direction[2]
                - center_to_contact[2] * desired_contact_direction[1]
            )
            torque_axis[1] = (
                center_to_contact[2] * desired_contact_direction[0]
                - center_to_contact[0] * desired_contact_direction[2]
            )
            torque_axis[2] = (
                center_to_contact[0] * desired_contact_direction[1]
                - center_to_contact[1] * desired_contact_direction[0]
            )
            alignment_torque_world = alignment_torque_scale * magnitude * torque_axis
            sphere_torque_world += alignment_torque_world

        lever_arm = sucker_position - element_center
        reaction_torque_world = np.empty(3, dtype=np.float64)
        reaction_torque_world[0] = (
            lever_arm[1] * reaction_force[2] - lever_arm[2] * reaction_force[1]
        )
        reaction_torque_world[1] = (
            lever_arm[2] * reaction_force[0] - lever_arm[0] * reaction_force[2]
        )
        reaction_torque_world[2] = (
            lever_arm[0] * reaction_force[1] - lever_arm[1] * reaction_force[0]
        )
        reaction_torque_world -= alignment_torque_world

        rod_local_torque = np.empty(3, dtype=np.float64)
        for axis in range(3):
            rod_local_torque[axis] = (
                rod_directors[axis, 0, idx] * reaction_torque_world[0]
                + rod_directors[axis, 1, idx] * reaction_torque_world[1]
                + rod_directors[axis, 2, idx] * reaction_torque_world[2]
            )
        rod_external_torques[:, idx] += rod_local_torque

    sphere_external_forces[..., 0] += sphere_force

    sphere_local_torque = np.zeros(3, dtype=np.float64)
    for axis in range(3):
        sphere_local_torque[axis] = (
            sphere_directors[axis, 0, 0] * sphere_torque_world[0]
            + sphere_directors[axis, 1, 0] * sphere_torque_world[1]
            + sphere_directors[axis, 2, 0] * sphere_torque_world[2]
        )
    sphere_external_torques[:, 0] += sphere_local_torque
