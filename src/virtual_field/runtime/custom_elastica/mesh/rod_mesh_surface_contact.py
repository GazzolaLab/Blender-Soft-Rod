from typing import Type

from numba import njit
from elastica.contact_forces import NoContact
from elastica.contact_utils import (
    _node_to_element_velocity,
    _elements_to_nodes_inplace,
    _find_slipping_elements,
)
import numpy as np
from elastica._linalg import (
    _batch_dot,
    _batch_norm,
    _batch_product_k_ik_to_ik,
    _batch_cross,
    _batch_matvec,
    _batch_matrix_transpose,
)


from .mesh_contact_utils import (
    Grid,
    _batch_sphere_triangle_intersection_check
)
from .mesh_surface import MeshSurface


class RodMeshSurfaceContactGridMethod(NoContact):
    """
    This class is for applying contact forces between a rod and a mesh surface.

    Examples
    --------
    How to define contact between rod and mesh surface.

    >>> simulator.detect_contact_between(rod, mesh_surface).using(
    ...    RodMeshSurfaceContactGridMethod,
    ...    k=1e4,
    ...    nu=10,
    ...    grid=grid
    ... )

    """

    def __init__(
        self,
        k: float,
        nu: float,
        grid: Grid,
        surface_tol=1e-4,
    ):
        super(RodMeshSurfaceContactGridMethod, self).__init__()
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
        self.grid = grid
        self.surface_tol = surface_tol
        

    @property
    def _allowed_system_two(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [MeshSurface]

    def apply_contact(
        self,
        system_one,
        system_two,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between rod object and mesh surface object.

        Parameters
        ----------
        system_one: RodType
            Rod object.
        system_two: AllowedContactType
            mesh surface object.

        """
        (
            self.position_idx_array,
            self.face_idx_array,
            self.element_position,
        ) = self.grid.find_faces(position_collection=system_one.position_collection)

        return self.rod_mesh_contact(
            system_two.faces,
            system_two.face_normals,
            system_two.face_centers,
            self.element_position,
            system_one.director_collection,
            system_two.side_vectors,
            self.position_idx_array,
            self.face_idx_array,
            self.surface_tol,
            self.k,
            self.nu,
            system_one.radius,
            system_one.mass,
            system_one.velocity_collection,
            system_one.external_forces,
        )

    @staticmethod
    @njit(cache=True)
    def rod_mesh_contact(
        faces,
        face_normals,
        face_centers,
        element_position,
        element_directors,
        side_vectors,
        position_idx_array,
        face_idx_array,
        surface_tol,
        k,
        nu,
        radius,
        mass,
        velocity_collection,
        external_forces,
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

        # Damping force response due to velocity towards the plane
        element_velocity = _node_to_element_velocity(
            mass=mass, node_velocity_collection=velocity_collection
        )

        if len(face_idx_array) == 0:
            return (
                np.empty((0,), dtype=radius.dtype),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((3, 0), dtype=element_position.dtype),
                np.empty((3, 0), dtype=element_position.dtype),
                np.empty((3, 0), dtype=element_velocity.dtype),
            )

        element_position_contacts = element_position[:, position_idx_array]
        # element_directors_contacts = element_directors[:, :, position_idx_array]
        contact_face_centers = face_centers[:, face_idx_array]
        normals_on_elements = face_normals[:, face_idx_array]
        radius_contacts = radius[position_idx_array]
        element_velocity_contacts = element_velocity[:, position_idx_array]
        contact_faces = faces[:, :, face_idx_array]
        contact_faces_vertex_A = contact_faces[:, 0, :]
        contact_faces_vertex_B = contact_faces[:, 1, :]
        contact_faces_vertex_C = contact_faces[:, 2, :]
        contact_faces_side_vectors = side_vectors[:, :, face_idx_array]
        contact_faces_side_AB = contact_faces_side_vectors[:, 0, :]
        contact_faces_side_AC = contact_faces_side_vectors[:, 1, :]
        contact_faces_side_BC = contact_faces_side_vectors[:, 2, :]

        # intersection check
        # TODO replace sphere-triangle intersection with cylinder-triangle intersection
        (
            no_intersection_idx,
            distance_from_face_plane,
        ) = _batch_sphere_triangle_intersection_check(
            sphere_centers=element_position_contacts,
            sphere_radii=radius_contacts,
            triangle_centers=contact_face_centers,
            triangle_normals=normals_on_elements,
            triangle_vertices_A=contact_faces_vertex_A,
            triangle_vertices_B=contact_faces_vertex_B,
            triangle_vertices_C=contact_faces_vertex_C,
            triangle_side_AB=contact_faces_side_AB,
            triangle_side_AC=contact_faces_side_AC,
            triangle_side_BC=contact_faces_side_BC,
            surface_tol=surface_tol,
        )

        # Elastic force response due to penetration
        plane_penetration = np.minimum(distance_from_face_plane - radius_contacts, 0.0)
        elastic_force = -k * _batch_product_k_ik_to_ik(
            plane_penetration, normals_on_elements
        )

        normal_component_of_element_velocity = _batch_dot(
            normals_on_elements, element_velocity_contacts
        )
        damping_force = -nu * _batch_product_k_ik_to_ik(
            normal_component_of_element_velocity, normals_on_elements
        )

        # Compute total plane response force
        plane_response_force_contacts = elastic_force + damping_force

        # Check if the rod elements are in contact with plane.
        no_penetration_idx = np.where(
            (distance_from_face_plane - radius_contacts) > surface_tol
        )[0]
        # check if the distance to the closest point on the triangle is smaller than the plane_intersection_radius (we can modify this later for cylinder)

        # If rod element does not have any penetration with face plane, plane cannot apply response
        # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
        plane_response_force_contacts[..., no_penetration_idx] = 0.0

        # If rod element does not intersect with face, plane cannot apply response
        # force on the element. Thus lets set plane response force to 0.0 for the no contact points.
        plane_response_force_contacts[..., no_intersection_idx] = 0.0

        plane_response_forces = np.zeros_like(external_forces)
        for i in range(len(position_idx_array)):
            plane_response_forces[
                :, position_idx_array[i]
            ] += plane_response_force_contacts[:, i]
        # Update the external forces
        _elements_to_nodes_inplace(plane_response_forces, external_forces)

        return (
            _batch_norm(plane_response_force_contacts),
            no_penetration_idx,
            no_intersection_idx,
            normals_on_elements,
            element_position_contacts,
            element_velocity_contacts,
        )


class RodMeshSurfaceContactGridMethodWithAnisotropicFriction(
    RodMeshSurfaceContactGridMethod
):
    """
    This class is for applying contact forces between a rod and a mesh surface
    with anisotropic friction.

    Examples
    --------
    How to define contact between rod and mesh surface.

    >>> simulator.detect_contact_between(rod, mesh_surface).using(
    ...    RodMeshSurfaceContactGridMethod,
    ...    k=1e4,
    ...    nu=10,
    ...    static_mu_array=np.array([0, 0, 0]),
    ...    kinetic_mu_array=np.array([1, 1.5, 2]),
    ...    grid=grid,
    ... )

    """

    def __init__(
        self,
        k: float,
        nu: float,
        slip_velocity_tol: float,
        gamma: float,
        static_mu_array: np.ndarray,
        kinetic_mu_array: np.ndarray,
        grid:Grid,
        surface_tol=1e-4,
    ):
        """
        Parameters
        ----------
        k : float
            Contact spring constant.
        nu : float
            Contact damping constant.
        slip_velocity_tol: float
            Velocity tolerance to determine if the element is slipping or not.
        gamma: float
            Viscous damping coefficient.
        static_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] static friction coefficients.
        kinetic_mu_array: numpy.ndarray
            1D (3,) array containing data with 'float' type.
            [forward, backward, sideways] kinetic friction coefficients.
        grid : grid class instance
            Dimension of the mesh surface's grid.
        """
        RodMeshSurfaceContactGridMethod.__init__(
            self, k, nu, grid, surface_tol
        )
        self.slip_velocity_tol = slip_velocity_tol
        self.gamma = gamma
        (
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
        ) = static_mu_array
        (
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
        ) = kinetic_mu_array

    @property
    def _allowed_system_two(self) -> list[Type]:
        # Modify this list to include the allowed system types for contact
        return [MeshSurface]

    def apply_contact(
        self,
        system_one,
        system_two,
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces and torques between rod object and mesh surface object.

        Parameters
        ----------
        system_one: Rod
            Rod object.
        system_two: MeshSurface
            mesh surface object.

        """
        (
            self.plane_response_force_mag,
            self.no_penetration_idx,
            self.no_intersection_idx,
            self.normals_on_elements,
            self.element_position_contacts,
            self.element_velocity_contacts,
        ) = super().apply_contact(system_one,system_two,time)

        self.mesh_anisotropic_friction(
            self.plane_response_force_mag,
            self.no_penetration_idx,
            self.no_intersection_idx,
            self.normals_on_elements,
            self.position_idx_array,
            self.element_velocity_contacts,
            self.slip_velocity_tol,
            self.kinetic_mu_forward,
            self.kinetic_mu_backward,
            self.kinetic_mu_sideways,
            self.gamma,
            self.static_mu_forward,
            self.static_mu_backward,
            self.static_mu_sideways,
            system_one.radius,
            system_one.tangents,
            system_one.director_collection,
            system_one.omega_collection,
            system_one.external_forces,
            system_one.external_torques,
        )

    
    @staticmethod
    @njit(cache=True)
    def mesh_anisotropic_friction(
        plane_response_force_mag,
        no_penetration_idx,
        no_intersection_idx,
        normals_on_elements,
        position_idx_array,
        element_velocity_contacts,
        slip_velocity_tol,
        kinetic_mu_forward,
        kinetic_mu_backward,
        kinetic_mu_sideways,
        gamma,
        static_mu_forward,
        static_mu_backward,
        static_mu_sideways,
        radius,
        tangents,
        director_collection,
        omega_collection,
        external_forces,
        external_torques,
        ):

        if len(position_idx_array) == 0:
            return no_penetration_idx, no_intersection_idx


        # First compute component of rod tangent in plane. Because friction forces acts in plane not out of plane. Thus
        # axial direction has to be in plane, it cannot be out of plane. We are projecting rod element tangent vector in
        # to the plane. So friction forces can only be in plane forces and not out of plane.

        if len(position_idx_array) > 0:
            tangents_contacts = tangents[:, position_idx_array]
            radius_contacts = radius[position_idx_array]
            omega_collection_contacts = omega_collection[:, position_idx_array]
            director_collection_contacts = director_collection[:, :, position_idx_array]
            kinetic_mu_sideways_array = kinetic_mu_sideways * np.ones_like(
                position_idx_array
            )
        else:
            tangents_contacts = tangents
            radius_contacts = radius
            omega_collection_contacts = omega_collection
            director_collection_contacts = director_collection
            kinetic_mu_sideways_array = kinetic_mu_sideways * np.ones_like(radius)

        tangent_along_normal_direction = _batch_dot(normals_on_elements, tangents_contacts)
        tangent_perpendicular_to_normal_direction = (
            tangents_contacts
            - _batch_product_k_ik_to_ik(tangent_along_normal_direction, normals_on_elements)
        )

        tangent_perpendicular_to_normal_direction_mag = _batch_norm(
            tangent_perpendicular_to_normal_direction
        )

        # Normalize tangent_perpendicular_to_normal_direction. This is axial direction for plane. Here we are adding
        # small tolerance (1e-10) for normalization, in order to prevent division by 0.
        axial_direction = _batch_product_k_ik_to_ik(
            1 / (tangent_perpendicular_to_normal_direction_mag + 1e-14),
            tangent_perpendicular_to_normal_direction,
        )

        # first apply axial kinetic friction
        velocity_mag_along_axial_direction = _batch_dot(
            element_velocity_contacts, axial_direction
        )
        velocity_along_axial_direction = _batch_product_k_ik_to_ik(
            velocity_mag_along_axial_direction, axial_direction
        )

        # Friction forces depends on the direction of velocity, in other words sign
        # of the velocity vector.
        velocity_sign_along_axial_direction = np.sign(velocity_mag_along_axial_direction)
        # Check top for sign convention
        kinetic_mu = 0.5 * (
            kinetic_mu_forward * (1 + velocity_sign_along_axial_direction)
            + kinetic_mu_backward * (1 - velocity_sign_along_axial_direction)
        )
        # Call slip function to check if elements slipping or not
        slip_function_along_axial_direction = _find_slipping_elements(
            velocity_along_axial_direction, slip_velocity_tol
        )

        # Now rolling kinetic friction
        rolling_direction = _batch_cross(axial_direction, normals_on_elements)
        torque_arm = -_batch_product_k_ik_to_ik(radius_contacts, normals_on_elements)
        velocity_along_rolling_direction = _batch_dot(
            element_velocity_contacts, rolling_direction
        )
        velocity_sign_along_rolling_direction = np.sign(velocity_along_rolling_direction)

        directors_transpose_contacts = _batch_matrix_transpose(director_collection_contacts)
        # directors_transpose = _batch_matrix_transpose(director_collection)

        # w_rot = Q.T @ omega @ Q @ r
        rotation_velocity = _batch_matvec(
            directors_transpose_contacts,
            _batch_cross(
                omega_collection_contacts,
                _batch_matvec(director_collection_contacts, torque_arm),
            ),
        )
        rotation_velocity_along_rolling_direction = _batch_dot(
            rotation_velocity, rolling_direction
        )
        slip_velocity_mag_along_rolling_direction = (
            velocity_along_rolling_direction + rotation_velocity_along_rolling_direction
        )
        slip_velocity_along_rolling_direction = _batch_product_k_ik_to_ik(
            slip_velocity_mag_along_rolling_direction, rolling_direction
        )
        slip_function_along_rolling_direction = _find_slipping_elements(
            slip_velocity_along_rolling_direction, slip_velocity_tol
        )
        # Compute unitized total slip velocity vector. We will use this to distribute the weight of the rod in axial
        # and rolling directions.
        unitized_total_velocity = (
            slip_velocity_along_rolling_direction + velocity_along_axial_direction
        )
        unitized_total_velocity /= _batch_norm(unitized_total_velocity + 1e-14)
        # Apply kinetic friction in axial direction.
        kinetic_friction_force_along_axial_direction_contacts = -(
            (1.0 - slip_function_along_axial_direction)
            * kinetic_mu
            * plane_response_force_mag
            * _batch_dot(unitized_total_velocity, axial_direction)
            * axial_direction
        )
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
        kinetic_friction_force_along_axial_direction_contacts[..., no_penetration_idx] = 0.0
        kinetic_friction_force_along_axial_direction_contacts[
            ..., no_intersection_idx
        ] = 0.0

        # Apply kinetic friction in rolling direction.
        kinetic_friction_force_along_rolling_direction_contacts = -(
            (1.0 - slip_function_along_rolling_direction)
            * kinetic_mu_sideways_array
            * plane_response_force_mag
            * _batch_dot(unitized_total_velocity, rolling_direction)
            * rolling_direction
        )
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set kinetic friction force to 0.0 for the no contact points.
        kinetic_friction_force_along_rolling_direction_contacts[
            ..., no_penetration_idx
        ] = 0.0
        kinetic_friction_force_along_rolling_direction_contacts[
            ..., no_intersection_idx
        ] = 0.0

        # torque = Q @ r @ Fr
        kinetic_rolling_torque_contacts = _batch_matvec(
            director_collection_contacts,
            _batch_cross(
                torque_arm, kinetic_friction_force_along_rolling_direction_contacts
            ),
        )

        # now axial static friction

        # check top for sign convention
        static_mu = 0.5 * (
            static_mu_forward * (1 + velocity_sign_along_axial_direction)
            + static_mu_backward * (1 - velocity_sign_along_axial_direction)
        )
        max_friction_force = (
            slip_function_along_axial_direction * static_mu * plane_response_force_mag
        )
        # friction = min(mu N, gamma v)
        static_friction_force_along_axial_direction_contacts = -(
            np.minimum(np.fabs(gamma * velocity_along_axial_direction), max_friction_force)
            * velocity_sign_along_axial_direction
            * axial_direction
        )
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set static friction force to 0.0 for the no contact points.
        static_friction_force_along_axial_direction_contacts[..., no_penetration_idx] = 0.0
        static_friction_force_along_axial_direction_contacts[
            ..., no_intersection_idx
        ] = 0.0

        # now rolling static friction
        # there is some normal, tangent and rolling directions inconsitency from Elastica
        # total_torques = _batch_matvec(directors_transpose, (internal_torques + external_torques))[:,position_idx_array]

        # Elastica has opposite defs of tangents in interaction.h and rod.cpp
        # total_torques_along_axial_direction = _batch_dot(total_torques, axial_direction)

        # noslip_force = -(
        #     (
        #         radius[position_idx_array] * force_component_along_rolling_direction
        #         - 2.0 * total_torques_along_axial_direction
        #     )
        #     / 3.0
        #     / radius[position_idx_array]
        # )

        max_friction_force = (
            slip_function_along_rolling_direction
            * static_mu_sideways
            * plane_response_force_mag
        )
        # noslip_force_sign = np.sign(noslip_force)

        static_friction_force_along_rolling_direction_contacts = (
            np.minimum(
                np.fabs(gamma * velocity_along_rolling_direction), max_friction_force
            )
            * velocity_sign_along_rolling_direction
            * rolling_direction
        )
        # If rod element does not have any contact with plane, plane cannot apply friction
        # force on the element. Thus lets set plane static friction force to 0.0 for the no contact points.
        static_friction_force_along_rolling_direction_contacts[
            ..., no_penetration_idx
        ] = 0.0
        static_friction_force_along_rolling_direction_contacts[
            ..., no_intersection_idx
        ] = 0.0

        static_rolling_torque_contacts = _batch_matvec(
            director_collection_contacts,
            _batch_cross(
                torque_arm, static_friction_force_along_rolling_direction_contacts
            ),
        )

        # add up contribution from all contacts
        kinetic_friction_force_along_axial_direction = np.zeros_like(tangents)
        kinetic_friction_force_along_rolling_direction = np.zeros_like(tangents)
        kinetic_rolling_torque = np.zeros_like(tangents)
        static_friction_force_along_axial_direction = np.zeros_like(tangents)
        static_friction_force_along_rolling_direction = np.zeros_like(tangents)
        static_rolling_torque = np.zeros_like(tangents)

        for i in range(len(position_idx_array)):
            kinetic_friction_force_along_axial_direction[
                :, position_idx_array[i]
            ] += kinetic_friction_force_along_axial_direction_contacts[:, i]
            kinetic_friction_force_along_rolling_direction[
                :, position_idx_array[i]
            ] += kinetic_friction_force_along_rolling_direction_contacts[:, i]
            kinetic_rolling_torque[
                :, position_idx_array[i]
            ] += kinetic_rolling_torque_contacts[:, i]
            static_friction_force_along_axial_direction[
                :, position_idx_array[i]
            ] += static_friction_force_along_axial_direction_contacts[:, i]
            static_friction_force_along_rolling_direction[
                :, position_idx_array[i]
            ] += static_friction_force_along_rolling_direction_contacts[:, i]
            static_rolling_torque[
                :, position_idx_array[i]
            ] += static_rolling_torque_contacts[:, i]

        # apply all forces and torques
        _elements_to_nodes_inplace(
            kinetic_friction_force_along_axial_direction, external_forces
        )
        _elements_to_nodes_inplace(
            kinetic_friction_force_along_rolling_direction, external_forces
        )
        external_torques += kinetic_rolling_torque
        _elements_to_nodes_inplace(
            static_friction_force_along_axial_direction, external_forces
        )
        _elements_to_nodes_inplace(
            static_friction_force_along_rolling_direction, external_forces
        )
        external_torques += static_rolling_torque

        return no_penetration_idx, no_intersection_idx