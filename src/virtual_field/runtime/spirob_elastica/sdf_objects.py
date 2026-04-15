from __future__ import annotations

from collections import deque

import numpy as np
if not hasattr(np, "typing"):
    import numpy.typing as np_typing

    np.typing = np_typing  # type: ignore[attr-defined]
from elastica import NoForces
from elastica.typing import RigidBodyType, RodType
from numba import njit


class SDFTorus(NoForces):
    """Torus contact via signed-distance field and Hertzian normal response.

    The torus is axis-aligned with world Z as the symmetry axis.
    SDF(point) = sqrt((sqrt(x^2+y^2)-R)^2 + z^2) - r
    where:
    - R: major radius
    - r: minor radius
    """

    def __init__(
        self,
        center: np.ndarray,
        major_radius: float,
        minor_radius: float,
        recording_queue: deque,
        stiffness: float = 1.0e5,
        damping: float = 10.0,
        record_interval: float = 1.0 / 90.0,
    ) -> None:
        super().__init__()
        self.center = np.asarray(center, dtype=np.float64).reshape(3)
        self.major_radius = float(major_radius)
        self.minor_radius = float(minor_radius)
        self.stiffness = float(stiffness)
        self.damping = float(damping)
        self.recording_queue = recording_queue
        self.record_interval = float(record_interval)
        self._last_record_time = -1.0e9

    def apply_forces(
        self,
        system: "RodType | RigidBodyType",
        time: np.float64 = np.float64(0.0),
    ) -> None:
        positions = system.position_collection
        velocities = system.velocity_collection
        external_forces = system.external_forces
        radii = system.radius

        node_radii = _node_radii_from_element_radii(radii, positions.shape[1])

        contact_points = np.empty((3, positions.shape[1]), dtype=np.float64)
        contact_count = _apply_torus_contact(
            positions,
            velocities,
            external_forces,
            node_radii,
            self.center,
            self.major_radius,
            self.minor_radius,
            self.stiffness,
            self.damping,
            contact_points,
        )
        current_time = float(time)
        if (
            contact_count > 0
            and current_time - self._last_record_time >= self.record_interval
        ):
            for idx in range(contact_count):
                self.recording_queue.append(
                    (
                        current_time,
                        [
                            float(contact_points[0, idx]),
                            float(contact_points[1, idx]),
                            float(contact_points[2, idx]),
                        ],
                    )
                )
            self._last_record_time = current_time


class SDFObstacleCylinders(NoForces):
    """Finite-cylinder contact via signed-distance fields and normal response."""

    def __init__(
        self,
        starts: np.ndarray,
        directions: np.ndarray,
        lengths: np.ndarray,
        radii: np.ndarray,
        normals: np.ndarray | None = None,
        stiffness: float = 8.0e4,
        damping: float = 4.0,
    ) -> None:
        super().__init__()
        self.starts = np.asarray(starts, dtype=np.float64)
        self.directions = np.asarray(directions, dtype=np.float64)
        self.lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
        self.radii = np.asarray(radii, dtype=np.float64).reshape(-1)
        if self.starts.ndim != 2 or self.starts.shape[1] != 3:
            raise ValueError("starts must have shape (n, 3)")
        if self.directions.shape != self.starts.shape:
            raise ValueError("directions must have the same shape as starts")
        if self.lengths.shape[0] != self.starts.shape[0]:
            raise ValueError("lengths must contain one entry per cylinder")
        if self.radii.shape[0] != self.starts.shape[0]:
            raise ValueError("radii must contain one entry per cylinder")
        if normals is None:
            self.normals = np.zeros_like(self.starts)
            self.normals[:, 0] = 1.0
        else:
            self.normals = np.asarray(normals, dtype=np.float64)
            if self.normals.shape != self.starts.shape:
                raise ValueError("normals must have the same shape as starts")
        self.stiffness = float(stiffness)
        self.damping = float(damping)

    def apply_forces(
        self,
        system: "RodType | RigidBodyType",
        time: np.float64 = np.float64(0.0),
    ) -> None:
        del time
        positions = system.position_collection
        velocities = system.velocity_collection
        external_forces = system.external_forces
        radii = system.radius

        node_radii = _node_radii_from_element_radii(radii, positions.shape[1])
        _apply_cylinder_contact(
            positions,
            velocities,
            external_forces,
            node_radii,
            self.starts,
            self.directions,
            self.lengths,
            self.radii,
            self.normals,
            self.stiffness,
            self.damping,
        )


def _node_radii_from_element_radii(
    element_radii: np.ndarray, n_nodes: int
) -> np.ndarray:
    radii = np.asarray(element_radii, dtype=np.float64).reshape(-1)
    if radii.size == n_nodes:
        return radii
    if radii.size != n_nodes - 1:
        raise ValueError(
            f"radius length must be n_nodes or n_nodes-1, got {radii.size} for n_nodes={n_nodes}"
        )

    node_radii = np.empty((n_nodes,), dtype=np.float64)
    node_radii[0] = radii[0]
    node_radii[-1] = radii[-1]
    node_radii[1:-1] = 0.5 * (radii[:-1] + radii[1:])
    return node_radii


def capped_cylinder_sdf_and_normal(
    point: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    fallback_normal: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    point_vec = np.asarray(point, dtype=np.float64).reshape(3)
    start_vec = np.asarray(start, dtype=np.float64).reshape(3)
    direction_vec = np.asarray(direction, dtype=np.float64).reshape(3)
    fallback = (
        np.asarray(fallback_normal, dtype=np.float64).reshape(3)
        if fallback_normal is not None
        else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    sdf, normal = _capped_cylinder_sdf_and_normal_impl(
        point_vec, start_vec, direction_vec, float(length), float(radius), fallback
    )
    return float(sdf), normal


def capped_cylinder_sdf(
    point: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
) -> float:
    sdf, _ = capped_cylinder_sdf_and_normal(point, start, direction, length, radius)
    return sdf


def obstacle_cylinder_sdf_and_normal(
    point: np.ndarray,
    starts: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    normals: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    starts_array = np.asarray(starts, dtype=np.float64)
    directions_array = np.asarray(directions, dtype=np.float64)
    lengths_array = np.asarray(lengths, dtype=np.float64).reshape(-1)
    radii_array = np.asarray(radii, dtype=np.float64).reshape(-1)
    if normals is None:
        normals_array = np.zeros_like(starts_array)
        normals_array[:, 0] = 1.0
    else:
        normals_array = np.asarray(normals, dtype=np.float64)
    point_vec = np.asarray(point, dtype=np.float64).reshape(3)
    sdf, normal, obstacle_index = _obstacle_cylinder_sdf_and_normal_impl(
        point_vec,
        starts_array,
        directions_array,
        lengths_array,
        radii_array,
        normals_array,
    )
    return float(sdf), normal, int(obstacle_index)


def _capped_cylinder_sdf_and_normal_impl(
    point: np.ndarray,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    fallback_normal: np.ndarray,
) -> tuple[float, np.ndarray]:
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    fallback_normal = np.asarray(fallback_normal, dtype=np.float64)

    eps = 1.0e-12
    rel = point - start
    axial = float(np.dot(rel, direction))
    radial = rel - axial * direction
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm > eps:
        radial_dir = radial / radial_norm
    else:
        fallback_norm = float(np.linalg.norm(fallback_normal))
        if fallback_norm > eps:
            radial_dir = fallback_normal / fallback_norm
        else:
            radial_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    if 0.0 <= axial <= length:
        if radial_norm >= radius:
            return radial_norm - radius, radial_dir
        side_distance = radius - radial_norm
        start_cap_distance = axial
        end_cap_distance = length - axial
        if side_distance <= start_cap_distance and side_distance <= end_cap_distance:
            return -(side_distance), radial_dir
        if start_cap_distance <= end_cap_distance:
            return -(start_cap_distance), -direction
        return -(end_cap_distance), direction

    cap_center = start if axial < 0.0 else start + direction * length
    cap_normal = -direction if axial < 0.0 else direction
    axial_out = -axial if axial < 0.0 else axial - length

    if radial_norm <= radius:
        return axial_out, cap_normal

    rim_point = cap_center + radial_dir * radius
    diff = point - rim_point
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm <= eps:
        return 0.0, cap_normal
    return diff_norm, diff / diff_norm


def _obstacle_cylinder_sdf_and_normal_impl(
    point: np.ndarray,
    starts: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    normals: np.ndarray,
) -> tuple[float, np.ndarray, int]:
    best_sdf = np.inf
    best_normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    best_index = -1
    for idx in range(starts.shape[0]):
        sdf, normal = _capped_cylinder_sdf_and_normal_impl(
            point,
            starts[idx],
            directions[idx],
            float(lengths[idx]),
            float(radii[idx]),
            normals[idx],
        )
        if sdf < best_sdf:
            best_sdf = sdf
            best_normal = normal
            best_index = idx
    return float(best_sdf), best_normal, best_index


@njit(cache=True)  # type: ignore
def _apply_torus_contact(
    positions: np.ndarray,
    velocities: np.ndarray,
    external_forces: np.ndarray,
    node_radii: np.ndarray,
    center: np.ndarray,
    major_radius: float,
    minor_radius: float,
    stiffness: float,
    damping: float,
    contact_points: np.ndarray,
) -> int:
    eps = 1.0e-12
    n_nodes = positions.shape[1]
    contact_count = 0

    for i in range(n_nodes):
        px = positions[0, i] - center[0]
        py = positions[1, i] - center[1]
        pz = positions[2, i] - center[2]

        q = np.sqrt(px * px + py * py)
        a = q - major_radius
        k = np.sqrt(a * a + pz * pz)
        sdf = k - minor_radius

        penetration = node_radii[i] - sdf
        if penetration <= 0.0:
            continue

        if q > eps:
            dq_dx = px / q
            dq_dy = py / q
        else:
            dq_dx = 1.0
            dq_dy = 0.0

        if k > eps:
            da_dk = a / k
            grad_x = da_dk * dq_dx
            grad_y = da_dk * dq_dy
            grad_z = pz / k
        else:
            grad_x = 0.0
            grad_y = 0.0
            grad_z = 1.0

        grad_norm = np.sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z)
        if grad_norm <= eps:
            continue

        nx = grad_x / grad_norm
        ny = grad_y / grad_norm
        nz = grad_z / grad_norm

        vx = velocities[0, i]
        vy = velocities[1, i]
        vz = velocities[2, i]
        vn = vx * nx + vy * ny + vz * nz

        # Hertzian normal contact + linear normal damping.
        force_mag = stiffness * penetration * np.sqrt(penetration) - damping * vn
        if force_mag <= 0.0:
            continue

        external_forces[0, i] += force_mag * nx
        external_forces[1, i] += force_mag * ny
        external_forces[2, i] += force_mag * nz

        # Closest point on torus surface for the current node.
        contact_points[0, contact_count] = positions[0, i] - sdf * nx
        contact_points[1, contact_count] = positions[1, i] - sdf * ny
        contact_points[2, contact_count] = positions[2, i] - sdf * nz
        contact_count += 1

    return contact_count


@njit(cache=True)  # type: ignore
def _apply_cylinder_contact(
    positions: np.ndarray,
    velocities: np.ndarray,
    external_forces: np.ndarray,
    node_radii: np.ndarray,
    starts: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    radii: np.ndarray,
    normals: np.ndarray,
    stiffness: float,
    damping: float,
) -> None:
    eps = 1.0e-12
    n_nodes = positions.shape[1]
    n_obstacles = starts.shape[0]

    for i in range(n_nodes):
        px = positions[0, i]
        py = positions[1, i]
        pz = positions[2, i]

        best_sdf = 1.0e30
        best_nx = 1.0
        best_ny = 0.0
        best_nz = 0.0

        for j in range(n_obstacles):
            sx = starts[j, 0]
            sy = starts[j, 1]
            sz = starts[j, 2]
            dx = directions[j, 0]
            dy = directions[j, 1]
            dz = directions[j, 2]
            length = lengths[j]
            radius = radii[j]
            fnx = normals[j, 0]
            fny = normals[j, 1]
            fnz = normals[j, 2]

            rx = px - sx
            ry = py - sy
            rz = pz - sz

            axial = rx * dx + ry * dy + rz * dz
            radial_x = rx - axial * dx
            radial_y = ry - axial * dy
            radial_z = rz - axial * dz
            radial_norm = np.sqrt(
                radial_x * radial_x + radial_y * radial_y + radial_z * radial_z
            )

            if radial_norm > eps:
                radial_dir_x = radial_x / radial_norm
                radial_dir_y = radial_y / radial_norm
                radial_dir_z = radial_z / radial_norm
            else:
                fallback_norm = np.sqrt(fnx * fnx + fny * fny + fnz * fnz)
                if fallback_norm > eps:
                    radial_dir_x = fnx / fallback_norm
                    radial_dir_y = fny / fallback_norm
                    radial_dir_z = fnz / fallback_norm
                else:
                    radial_dir_x = 1.0
                    radial_dir_y = 0.0
                    radial_dir_z = 0.0

            sdf = 0.0
            nx = 0.0
            ny = 0.0
            nz = 0.0

            if 0.0 <= axial <= length:
                if radial_norm >= radius:
                    sdf = radial_norm - radius
                    nx = radial_dir_x
                    ny = radial_dir_y
                    nz = radial_dir_z
                else:
                    side_distance = radius - radial_norm
                    start_cap_distance = axial
                    end_cap_distance = length - axial
                    if (
                        side_distance <= start_cap_distance
                        and side_distance <= end_cap_distance
                    ):
                        sdf = -side_distance
                        nx = radial_dir_x
                        ny = radial_dir_y
                        nz = radial_dir_z
                    elif start_cap_distance <= end_cap_distance:
                        sdf = -start_cap_distance
                        nx = -dx
                        ny = -dy
                        nz = -dz
                    else:
                        sdf = -end_cap_distance
                        nx = dx
                        ny = dy
                        nz = dz
            else:
                if axial < 0.0:
                    cap_center_x = sx
                    cap_center_y = sy
                    cap_center_z = sz
                    cap_normal_x = -dx
                    cap_normal_y = -dy
                    cap_normal_z = -dz
                    axial_out = -axial
                else:
                    cap_center_x = sx + dx * length
                    cap_center_y = sy + dy * length
                    cap_center_z = sz + dz * length
                    cap_normal_x = dx
                    cap_normal_y = dy
                    cap_normal_z = dz
                    axial_out = axial - length

                if radial_norm <= radius:
                    sdf = axial_out
                    nx = cap_normal_x
                    ny = cap_normal_y
                    nz = cap_normal_z
                else:
                    rim_x = cap_center_x + radial_dir_x * radius
                    rim_y = cap_center_y + radial_dir_y * radius
                    rim_z = cap_center_z + radial_dir_z * radius
                    diff_x = px - rim_x
                    diff_y = py - rim_y
                    diff_z = pz - rim_z
                    diff_norm = np.sqrt(
                        diff_x * diff_x + diff_y * diff_y + diff_z * diff_z
                    )
                    if diff_norm <= eps:
                        sdf = 0.0
                        nx = cap_normal_x
                        ny = cap_normal_y
                        nz = cap_normal_z
                    else:
                        sdf = diff_norm
                        nx = diff_x / diff_norm
                        ny = diff_y / diff_norm
                        nz = diff_z / diff_norm

            if sdf < best_sdf:
                best_sdf = sdf
                best_nx = nx
                best_ny = ny
                best_nz = nz

        penetration = node_radii[i] - best_sdf
        if penetration <= 0.0:
            continue

        vx = velocities[0, i]
        vy = velocities[1, i]
        vz = velocities[2, i]
        vn = vx * best_nx + vy * best_ny + vz * best_nz

        force_mag = stiffness * penetration * np.sqrt(penetration) - damping * vn
        if force_mag <= 0.0:
            continue

        external_forces[0, i] += force_mag * best_nx
        external_forces[1, i] += force_mag * best_ny
        external_forces[2, i] += force_mag * best_nz
