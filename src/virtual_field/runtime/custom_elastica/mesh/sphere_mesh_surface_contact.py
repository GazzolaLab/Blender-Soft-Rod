from typing import Type

import numpy as np
from elastica._linalg import (
    _batch_norm,
    _batch_product_i_ik_to_k,
    _batch_product_k_ik_to_ik,
)
from elastica.contact_forces import NoContact, common_check_systems_validity
from elastica.rigidbody.sphere import Sphere
from numba import njit

from .mesh_contact_utils import _batch_sphere_triangle_intersection_check
from .mesh_surface import MeshSurface


class SphereMeshSurfaceContact(NoContact):
    """
    This class is for applying contact forces between a sphere and a mesh surface.

    Examples
    --------
    How to define contact between sphere and mesh surface.

    >>> simulator.detect_contact_between(sphere, mesh_surface).using(
    ...    SphereMeshSurfaceContact,
    ...    k=1e4,
    ...    nu=10,
    ...    search_radius = 1,
    ... )

    """

    def __init__(
        self,
        k: float,
        nu: float,
        search_radius: float,
        surface_tol=1e-4,
    ):
        super(SphereMeshSurfaceContact, self).__init__()
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        grid : grid class instance
            Dimension of the mesh surface's grid.
        """
        self.k = k
        self.nu = nu
        self.search_radius = search_radius
        self.surface_tol = surface_tol

    @property
    def _allowed_system_one(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [Sphere]

    @property
    def _allowed_system_two(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [MeshSurface]

    def _check_systems_validity(
        self,
        system_one,
        system_two,
    ) -> None:
        """
        This checks the contact order and type of a sphere object and a mesh surface object.
        For the SphereMeshSurfaceContact class system_one should be a sphere and system_two should be a mesh surface.
        This also checks if search radius is chosen appropriately.
        """
        common_check_systems_validity(system_one, self._allowed_system_one)
        common_check_systems_validity(system_two, self._allowed_system_two)
        if system_one.radius > self.search_radius:
            raise ValueError(
                f"Search radius must be greater than the sphere radius."
            )

    def apply_contact(
        self,
        system_one,
        system_two,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between sphere object and mesh surface object.

        """
        self.face_idx_array = self.search_faces(
            self.search_radius,
            system_one.position_collection,
            system_two.faces[:, 0, :],
            system_two.faces[:, 1, :],
            system_two.faces[:, 2, :],
        )

        return self.sphere_mesh_contact(
            system_one.position_collection,
            system_one.radius,
            system_one.velocity_collection,
            system_one.external_forces,
            system_two.faces,
            system_two.face_normals,
            system_two.face_centers,
            system_two.side_vectors,
            self.face_idx_array,
            self.surface_tol,
            self.k,
            self.nu,
        )

    @staticmethod
    @njit(cache=True)
    def search_faces(
        search_radius, position, faces_vertex_A, faces_vertex_B, faces_vertex_C
    ):
        idx_A = _batch_norm(faces_vertex_A - position) < search_radius
        idx_B = _batch_norm(faces_vertex_B - position) < search_radius
        idx_C = _batch_norm(faces_vertex_C - position) < search_radius
        return np.where(idx_A + idx_B + idx_C)[0]

    @staticmethod
    @njit(cache=True)
    def sphere_mesh_contact(
        position,
        radius,
        velocity,
        external_forces,
        faces,
        face_normals,
        face_centers,
        side_vectors,
        face_idx_array,
        surface_tol,
        k,
        nu,
    ):
        """
        This function computes the plane force response on the element, in the
        case of contact. Contact model given in Eqn 4.8 Gazzola et. al. RSoS 2018 paper
        is used.

        Parameters
        ----------
        system

        Returns
        -------
        magnitude of the plane response
        """

        if len(face_idx_array) == 0:
            return (
                np.empty((0,), dtype=position.dtype),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((3, 0), dtype=position.dtype),
            )

        contact_face_centers = face_centers[:, face_idx_array]
        contact_face_normals = face_normals[:, face_idx_array]
        contact_faces = faces[:, :, face_idx_array]
        contact_faces_vertex_A = contact_faces[:, 0, :]
        contact_faces_vertex_B = contact_faces[:, 1, :]
        contact_faces_vertex_C = contact_faces[:, 2, :]
        contact_faces_side_vectors = side_vectors[:, :, face_idx_array]
        contact_faces_side_AB = contact_faces_side_vectors[:, 0, :]
        contact_faces_side_AC = contact_faces_side_vectors[:, 1, :]
        contact_faces_side_BC = contact_faces_side_vectors[:, 2, :]

        (
            no_intersection_idx,
            distance_from_face_plane,
        ) = _batch_sphere_triangle_intersection_check(
            sphere_centers=position,
            sphere_radii=radius,
            triangle_centers=contact_face_centers,
            triangle_normals=contact_face_normals,
            triangle_vertices_A=contact_faces_vertex_A,
            triangle_vertices_B=contact_faces_vertex_B,
            triangle_vertices_C=contact_faces_vertex_C,
            triangle_side_AB=contact_faces_side_AB,
            triangle_side_AC=contact_faces_side_AC,
            triangle_side_BC=contact_faces_side_BC,
            surface_tol=surface_tol,
        )

        # Elastic force response due to penetration
        plane_penetration = np.minimum(distance_from_face_plane - radius, 0.0)
        elastic_force = -k * _batch_product_k_ik_to_ik(
            plane_penetration, contact_face_normals
        )

        normal_component_of_element_velocity = _batch_product_i_ik_to_k(
            velocity[:, 0], contact_face_normals
        )
        damping_force = -nu * _batch_product_k_ik_to_ik(
            normal_component_of_element_velocity, contact_face_normals
        )

        # Compute total plane response force
        plane_response_force_contacts = elastic_force + damping_force

        # Check if the sphere are in contact with plane.
        no_penetration_idx = np.where(
            (distance_from_face_plane - radius) > surface_tol
        )[0]
        # check if the distance to the closest point on the triangle is smaller than the plane_intersection_radius

        # If sphere does not have any penetration with face plane, plane cannot apply response
        # force on the sphere. Thus lets set plane response force to 0.0 for the no contact points.
        plane_response_force_contacts[..., no_penetration_idx] = 0.0

        # If sphere does not intersect with face, plane cannot apply response
        # force on the sphere. Thus lets set plane response force to 0.0 for the no contact points.
        plane_response_force_contacts[..., no_intersection_idx] = 0.0

        # Update the external forces
        external_forces[:, 0] += np.sum(plane_response_force_contacts, axis=1)

        return (
            _batch_norm(plane_response_force_contacts),
            no_penetration_idx,
            no_intersection_idx,
            contact_face_normals,
        )
