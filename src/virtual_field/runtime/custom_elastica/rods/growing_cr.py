from typing import Type
from typing_extensions import Self

from numpy.typing import NDArray

import numpy as np
from numpy.testing import assert_allclose
import elastica as ea
from elastica.utils import Tolerance
from elastica.rod import RodBase
from elastica.rod.factory_function import (
    _assert_dim,
    _directors_validity_checker,
)
from elastica.systems.protocol import SystemProtocol
from elastica._linalg import (
    _batch_cross,
    _batch_norm,
    _batch_dot,
)
from elastica.rod.cosserat_rod import (
    _compute_bending_twist_strains,
    _compute_internal_forces,
    _compute_internal_torques,
    _compute_shear_stretch_strains,
    _update_accelerations,
    _zeroed_out_external_forces_and_torques,
)


def growing_cr_allocate(
    total_elements: int,
    min_elements: int,
    current_elements: int,
    base_position: np.ndarray,
    direction: NDArray[np.float64],
    normal: NDArray[np.float64],
    base_length: np.float64,
    base_radius: np.float64,
    density: np.float64,
    youngs_modulus: np.float64,
):
    n_elements = total_elements
    n_nodes = n_elements + 1
    n_voronoi_elements = n_elements - 1

    position = np.zeros((3, n_nodes))
    elemental_length = base_length / total_elements
    start = base_position - direction * elemental_length * (
        total_elements - current_elements
    )
    end = base_position + direction * elemental_length * current_elements
    for i in range(3):
        position[i, ...] = np.linspace(start[i], end[i], n_elements + 1)

    # Compute rest lengths and tangents
    position_diff = position[..., 1:] - position[..., :-1]
    rest_lengths = _batch_norm(position_diff)
    tangents = position_diff / rest_lengths
    normal /= np.linalg.norm(normal)

    directors = np.zeros((3, 3, n_elements))
    normal_collection = np.repeat(normal[:, np.newaxis], n_elements, axis=1)
    assert_allclose(
        _batch_dot(normal_collection, tangents),
        0,
        atol=Tolerance.atol(),
        err_msg=(" Rod normal and tangent are not perpendicular to each other!"),
    )
    directors[0, ...] = normal_collection
    directors[1, ...] = _batch_cross(tangents, normal_collection)
    directors[2, ...] = tangents
    _directors_validity_checker(directors, tangents, n_elements)

    # Set radius array
    radius = np.zeros((n_elements))
    # Check if the user input radius is valid
    radius_temp = np.array(base_radius)
    _assert_dim(radius_temp, 2, "radius")
    radius[:] = radius_temp
    # Check if the elements of radius are greater than tolerance
    assert np.all(radius > Tolerance.atol()), " Radius has to be greater than 0."

    # Set density array
    density_array = np.zeros((n_elements))
    # Check if the user input density is valid
    density_temp = np.array(density)
    _assert_dim(density_temp, 2, "density")
    density_array[:] = density_temp
    # Check if the elements of density are greater than tolerance
    assert np.all(density_array > Tolerance.atol()), (
        " Density has to be greater than 0."
    )

    # Second moment of inertia
    A0 = np.pi * radius * radius
    I0_1 = A0 * A0 / (4.0 * np.pi)
    I0_2 = I0_1
    I0_3 = 2.0 * I0_2
    I0 = np.array([I0_1, I0_2, I0_3]).transpose()
    # Mass second moment of inertia for disk cross-section
    mass_second_moment_of_inertia = np.zeros((3, 3, n_elements), np.float64)

    mass_second_moment_of_inertia_temp = np.einsum(
        "ij,i->ij", I0, density * rest_lengths
    )

    for i in range(n_elements):
        np.fill_diagonal(
            mass_second_moment_of_inertia[..., i],
            mass_second_moment_of_inertia_temp[i, :],
        )

    # Inverse of second moment of inertia
    inv_mass_second_moment_of_inertia = np.zeros((3, 3, n_elements))
    for i in range(n_elements):
        # Check rank of mass moment of inertia matrix to see if it is invertible
        assert np.linalg.matrix_rank(mass_second_moment_of_inertia[..., i]) == 3
        inv_mass_second_moment_of_inertia[..., i] = np.linalg.inv(
            mass_second_moment_of_inertia[..., i]
        )

    # Shear/Stretch matrix
    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.5))

    # Value taken based on best correlation for Poisson ratio = 0.5, from
    # "On Timoshenko's correction for shear in vibrating beams" by Kaneko, 1975
    alpha_c = 27.0 / 28.0
    shear_matrix = np.zeros((3, 3, n_elements), np.float64)
    for i in range(n_elements):
        np.fill_diagonal(
            shear_matrix[..., i],
            [
                alpha_c * shear_modulus * A0[i],
                alpha_c * shear_modulus * A0[i],
                youngs_modulus * A0[i],
            ],
        )

    # Bend/Twist matrix
    bend_matrix = np.zeros((3, 3, n_voronoi_elements + 1), np.float64)
    for i in range(n_elements):
        np.fill_diagonal(
            bend_matrix[..., i],
            [
                youngs_modulus * I0_1[i],
                youngs_modulus * I0_2[i],
                shear_modulus * I0_3[i],
            ],
        )
    for i in range(0, 3):
        assert np.all(bend_matrix[i, i, :] > Tolerance.atol()), (
            " Bend matrix has to be greater than 0."
        )

    # Compute bend matrix in Voronoi Domain
    rest_lengths_temp_for_voronoi = rest_lengths
    bend_matrix = (
        bend_matrix[..., 1:] * rest_lengths_temp_for_voronoi[1:]
        + bend_matrix[..., :-1] * rest_lengths_temp_for_voronoi[0:-1]
    ) / (rest_lengths_temp_for_voronoi[1:] + rest_lengths_temp_for_voronoi[:-1])

    # Compute volume of elements
    volume = np.pi * radius**2 * rest_lengths

    # Compute mass of elements
    mass = np.zeros(n_nodes)
    mass[:-1] += 0.5 * density * volume
    mass[1:] += 0.5 * density * volume

    rest_sigma = np.zeros((3, n_elements))
    rest_kappa = np.zeros((3, n_voronoi_elements))

    # Compute rest voronoi length
    rest_voronoi_lengths = 0.5 * (
        rest_lengths_temp_for_voronoi[1:] + rest_lengths_temp_for_voronoi[:-1]
    )

    # Allocate arrays for Cosserat Rod equations
    velocities = np.zeros((3, n_nodes))
    omegas = np.zeros((3, n_elements))
    accelerations = 0.0 * velocities
    angular_accelerations = 0.0 * omegas

    internal_forces = 0.0 * accelerations
    internal_torques = 0.0 * angular_accelerations

    external_forces = 0.0 * accelerations
    external_torques = 0.0 * angular_accelerations

    lengths = np.zeros((n_elements))
    tangents = np.zeros((3, n_elements))

    dilatation = np.zeros((n_elements))
    voronoi_dilatation = np.zeros((n_voronoi_elements))
    dilatation_rate = np.zeros((n_elements))

    sigma = np.zeros((3, n_elements))
    kappa = np.zeros((3, n_voronoi_elements))

    internal_stress = np.zeros((3, n_elements))
    internal_couple = np.zeros((3, n_voronoi_elements))

    return (
        n_elements,
        position,
        velocities,
        omegas,
        accelerations,
        angular_accelerations,
        directors,
        radius,
        mass_second_moment_of_inertia,
        inv_mass_second_moment_of_inertia,
        shear_matrix,
        bend_matrix,
        density_array,
        volume,
        mass,
        internal_forces,
        internal_torques,
        external_forces,
        external_torques,
        lengths,
        rest_lengths,
        tangents,
        dilatation,
        dilatation_rate,
        voronoi_dilatation,
        rest_voronoi_lengths,
        sigma,
        kappa,
        rest_sigma,
        rest_kappa,
        internal_stress,
        internal_couple,
    )


class GrowingCR(ea.CosseratRod, SystemProtocol):
    """
    Cosserat rod with fixed-capacity storage (``total_elements``) and a configurable
    active suffix of ``current_elements`` elements. Internal physics and acceleration
    updates use only the active tail (equivalent to slicing with ``[-current_elements:]``
    on per-element arrays and the last ``current_elements + 1`` nodes).
    """

    REQUISITE_MODULES: list[Type] = []

    def __init__(
        self,
        total_elements: int,
        current_elements: int,
        min_elements: int,
        position: NDArray[np.float64],
        velocity: NDArray[np.float64],
        omega: NDArray[np.float64],
        acceleration: NDArray[np.float64],
        angular_acceleration: NDArray[np.float64],
        directors: NDArray[np.float64],
        radius: NDArray[np.float64],
        mass_second_moment_of_inertia: NDArray[np.float64],
        inv_mass_second_moment_of_inertia: NDArray[np.float64],
        shear_matrix: NDArray[np.float64],
        bend_matrix: NDArray[np.float64],
        density_array: NDArray[np.float64],
        volume: NDArray[np.float64],
        mass: NDArray[np.float64],
        internal_forces: NDArray[np.float64],
        internal_torques: NDArray[np.float64],
        external_forces: NDArray[np.float64],
        external_torques: NDArray[np.float64],
        lengths: NDArray[np.float64],
        rest_lengths: NDArray[np.float64],
        tangents: NDArray[np.float64],
        dilatation: NDArray[np.float64],
        dilatation_rate: NDArray[np.float64],
        voronoi_dilatation: NDArray[np.float64],
        rest_voronoi_lengths: NDArray[np.float64],
        sigma: NDArray[np.float64],
        kappa: NDArray[np.float64],
        rest_sigma: NDArray[np.float64],
        rest_kappa: NDArray[np.float64],
        internal_stress: NDArray[np.float64],
        internal_couple: NDArray[np.float64],
    ) -> None:
        self.total_elements = total_elements
        self.current_elements = current_elements
        self.min_elements = min_elements
        self.n_elems = total_elements
        self.n_nodes = total_elements + 1
        self.ring_rod_flag = False

        self.position_collection = position
        self.velocity_collection = velocity
        self.omega_collection = omega
        self.acceleration_collection = acceleration
        self.alpha_collection = angular_acceleration
        self.director_collection = directors
        self.radius = radius
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia
        self.inv_mass_second_moment_of_inertia = inv_mass_second_moment_of_inertia
        self.shear_matrix = shear_matrix
        self.bend_matrix = bend_matrix
        self.density = density_array
        self.volume = volume
        self.mass = mass
        self.internal_forces = internal_forces
        self.internal_torques = internal_torques
        self.external_forces = external_forces
        self.external_torques = external_torques
        self.lengths = lengths
        self.rest_lengths = rest_lengths
        self.tangents = tangents
        self.dilatation = dilatation
        self.dilatation_rate = dilatation_rate
        self.voronoi_dilatation = voronoi_dilatation
        self.rest_voronoi_lengths = rest_voronoi_lengths
        self.sigma = sigma
        self.kappa = kappa
        self.rest_sigma = rest_sigma
        self.rest_kappa = rest_kappa
        self.internal_stress = internal_stress
        self.internal_couple = internal_couple

        self.ghost_elems_idx = np.array([], dtype=np.int32)
        self.ghost_voronoi_idx = np.array([], dtype=np.int32)

        ns = self._active_node_slice()
        es = self._active_elem_slice()
        vs = self._active_voronoi_slice()

        _compute_shear_stretch_strains(
            self.position_collection[:, ns],
            self.volume[es],
            self.lengths[es],
            self.tangents[:, es],
            self.radius[es],
            self.rest_lengths[es],
            self.rest_voronoi_lengths[vs],
            self.dilatation[es],
            self.voronoi_dilatation[vs],
            self.director_collection[:, :, es],
            self.sigma[:, es],
        )

        _compute_bending_twist_strains(
            self.director_collection[:, :, es],
            self.rest_voronoi_lengths[vs],
            self.kappa[:, vs],
        )

    # @property
    # def base_position(self) -> NDArray[np.float64]:
    #     return self.position_collection[:, self.current_elements]

    # @property
    # def base_orientation(self) -> NDArray[np.float64]:
    #     return self.director_collection[:, :, self.current_elements]

    def _active_node_slice(self) -> slice:
        te, ce = self.total_elements, self.current_elements
        return slice(te - ce, te + 1)

    def _active_elem_slice(self) -> slice:
        te, ce = self.total_elements, self.current_elements
        return slice(te - ce, te)

    def _active_voronoi_slice(self) -> slice:
        te, ce = self.total_elements, self.current_elements
        return slice(te - ce, te - 1)

    def set_current_elements(self, n: int) -> None:
        """Resize the active suffix; updates ``n_elems`` / ``n_nodes`` for the active region."""
        if not (self.min_elements <= n <= self.total_elements):
            return
        self.current_elements = n
        self.n_elems = n
        self.n_nodes = n + 1

    @classmethod
    def straight_rod(
        cls,
        total_elements: int,
        min_elements: int,
        current_elements: int,
        base_position: NDArray[np.float64],
        direction: NDArray[np.float64],
        normal: NDArray[np.float64],
        base_length: float,
        base_radius: float,
        density: float,
        youngs_modulus: float,
    ) -> Self:
        """
        Cosserat rod constructor for straight-rod geometry.


        Notes
        -----
        Shear modulus follows Poisson's ratio 0.5 inside ``growing_cr_allocate``.
        """
        (
            total_elems,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
        ) = growing_cr_allocate(
            total_elements,
            min_elements,
            current_elements,
            base_position,
            direction,
            normal,
            np.float64(base_length),
            np.float64(base_radius),
            np.float64(density),
            np.float64(youngs_modulus),
        )
        if total_elems != total_elements:
            raise RuntimeError("growing_cr_allocate total element count mismatch")

        return cls(
            total_elements,
            current_elements,
            min_elements,
            position,
            velocities,
            omegas,
            accelerations,
            angular_accelerations,
            directors,
            radius,
            mass_second_moment_of_inertia,
            inv_mass_second_moment_of_inertia,
            shear_matrix,
            bend_matrix,
            density_array,
            volume,
            mass,
            internal_forces,
            internal_torques,
            external_forces,
            external_torques,
            lengths,
            rest_lengths,
            tangents,
            dilatation,
            dilatation_rate,
            voronoi_dilatation,
            rest_voronoi_lengths,
            sigma,
            kappa,
            rest_sigma,
            rest_kappa,
            internal_stress,
            internal_couple,
        )

    def compute_internal_forces_and_torques(self, time: np.float64) -> None:
        """
        Compute internal forces and torques. We need to compute internal forces and torques before the acceleration because
        they are used in interaction. Thus in order to speed up simulation, we will compute internal forces and torques
        one time and use them. Previously, we were computing internal forces and torques multiple times in interaction.
        Saving internal forces and torques in a variable take some memory, but we will gain speed up.

        Parameters
        ----------
        time: np.float64
            current time

        """
        ns = self._active_node_slice()
        es = self._active_elem_slice()
        vs = self._active_voronoi_slice()

        _compute_internal_forces(
            self.position_collection[:, ns],
            self.volume[es],
            self.lengths[es],
            self.tangents[:, es],
            self.radius[es],
            self.rest_lengths[es],
            self.rest_voronoi_lengths[vs],
            self.dilatation[es],
            self.voronoi_dilatation[vs],
            self.director_collection[:, :, es],
            self.sigma[:, es],
            self.rest_sigma[:, es],
            self.shear_matrix[:, :, es],
            self.internal_stress[:, es],
            self.internal_forces[:, ns],
            self.ghost_elems_idx,
        )

        _compute_internal_torques(
            self.position_collection[:, ns],
            self.velocity_collection[:, ns],
            self.tangents[:, es],
            self.lengths[es],
            self.rest_lengths[es],
            self.director_collection[:, :, es],
            self.rest_voronoi_lengths[vs],
            self.bend_matrix[:, :, vs],
            self.rest_kappa[:, vs],
            self.kappa[:, vs],
            self.voronoi_dilatation[vs],
            self.mass_second_moment_of_inertia[:, :, es],
            self.omega_collection[:, es],
            self.internal_stress[:, es],
            self.internal_couple[:, vs],
            self.dilatation[es],
            self.dilatation_rate[es],
            self.internal_torques[:, es],
            self.ghost_voronoi_idx,
        )

    # Interface to time-stepper mixins (Symplectic, Explicit), which calls this method
    def update_accelerations(self, time: np.float64, dt: np.float64) -> None:
        """
        Updates the acceleration variables

        Parameters
        ----------
        time: np.float64
            current time
        dt: np.float64
            timestep. (may involve designing implicit solver)

        """
        ns = self._active_node_slice()
        es = self._active_elem_slice()

        _update_accelerations(
            self.acceleration_collection[:, ns],
            self.internal_forces[:, ns],
            self.external_forces[:, ns],
            self.mass[ns],
            self.alpha_collection[:, es],
            self.inv_mass_second_moment_of_inertia[:, :, es],
            self.internal_torques[:, es],
            self.external_torques[:, es],
            self.dilatation[es],
        )

    def zeroed_out_external_forces_and_torques(self, time: np.float64) -> None:
        _zeroed_out_external_forces_and_torques(
            self.external_forces, self.external_torques
        )
