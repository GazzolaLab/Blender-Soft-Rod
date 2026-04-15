from numba import njit
import numpy as np
from elastica.contact_utils import (
    _node_to_element_position,
)
from elastica._linalg import (
    _batch_dot,
    _batch_norm,
)
from functools import partial
from elastica.typing import RodType
from .mesh_surface import MeshSurface
# import dill


class BoundaryError(Exception):
    "Raised when rod leaves surface grid boundary"


@njit(cache=True)
def _batch_sphere_triangle_intersection_check(
    sphere_centers,
    sphere_radii,
    triangle_centers,
    triangle_normals,
    triangle_vertices_A,
    triangle_vertices_B,
    triangle_vertices_C,
    triangle_side_AB,
    triangle_side_AC,
    triangle_side_BC,
    surface_tol,
):
    """
    This checks intersections between a set of spheres and corresponding sets of triangles
    ----------
    sphere_centers : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    sphere_radii : numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
    triangle_centers : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_normals : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_vertices_A : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_vertices_B : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_vertices_C : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_side_AB : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_side_AC : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    triangle_side_BC : numpy.ndarray
        2D (dim, blocksize) array containing data with 'float' type.
    surface_tol : float

    Returns
    -------
    no_intersection_idx : numpy.ndarray
        1D (blocksize) array containing data with 'bool' type.
    distance_from_triangle_plane : numpy.ndarray
        1D (blocksize) array containing data with 'float' type.
    Notes
    ----------
    based on Fedor's answer in https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
    Parameters

    """
    center_to_center_vector = sphere_centers - triangle_centers
    distance_from_triangle_plane = np.abs(
        _batch_dot(triangle_normals, center_to_center_vector)
    )
    projected_sphere_center = (
        sphere_centers - distance_from_triangle_plane * triangle_normals
    )
    triangle_intersection_radius = np.sqrt(
        np.maximum(sphere_radii**2 - distance_from_triangle_plane**2, 0)
    )  # using pythagoras, figure out the intersection radius
    closest_point_on_triangle_to_projected_point = projected_sphere_center.copy()

    projected_sphere_center_to_vertex_A = projected_sphere_center - triangle_vertices_A
    projected_sphere_center_to_vertex_B = projected_sphere_center - triangle_vertices_B
    projected_sphere_center_to_vertex_C = projected_sphere_center - triangle_vertices_C

    d1 = _batch_dot(projected_sphere_center_to_vertex_A, triangle_side_AB)
    d2 = _batch_dot(projected_sphere_center_to_vertex_A, triangle_side_AC)
    d3 = _batch_dot(projected_sphere_center_to_vertex_B, triangle_side_AB)
    d4 = _batch_dot(projected_sphere_center_to_vertex_B, triangle_side_AC)
    d5 = _batch_dot(projected_sphere_center_to_vertex_C, triangle_side_AB)
    d6 = _batch_dot(projected_sphere_center_to_vertex_C, triangle_side_AC)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    region_1 = np.where((d1 <= 0) * (d2 <= 0))[0]  # region 1 (closest to vertex A)
    closest_point_on_triangle_to_projected_point[:, region_1] = triangle_vertices_A[
        :, region_1
    ]
    region_2 = np.where((d3 >= 0) * (d4 <= d3))[0]  # region 2 (closest to vertex B)
    closest_point_on_triangle_to_projected_point[:, region_2] = triangle_vertices_B[
        :, region_2
    ]
    region_3 = np.where((d6 >= 0) * (d5 <= d6))[0]  # region 3 (closest to vertex C)
    closest_point_on_triangle_to_projected_point[:, region_3] = triangle_vertices_C[
        :, region_3
    ]
    region_4 = np.where((vc <= 0) * (d1 >= 0) * (d3 <= 0))[
        0
    ]  # region 4 (closest to some point on AB)
    closest_point_on_triangle_to_projected_point[:, region_4] = triangle_vertices_A[
        :, region_4
    ] + d1[region_4] * triangle_side_AB[:, region_4] / (d1[region_4] - d3[region_4])
    region_5 = np.where((vb <= 0) * (d2 >= 0) * (d6 <= 0))[
        0
    ]  # region 5 (closest to some point on AC)
    closest_point_on_triangle_to_projected_point[:, region_5] = triangle_vertices_A[
        :, region_5
    ] + d2[region_5] * triangle_side_AC[:, region_5] / (d2[region_5] - d6[region_5])
    region_6 = np.where((va <= 0) * (d4 >= d3) * (d5 >= d6))[
        0
    ]  # region 6 (closest to some point on BC)
    closest_point_on_triangle_to_projected_point[:, region_6] = triangle_vertices_B[
        :, region_6
    ] + (d4[region_6] - d3[region_6]) * triangle_side_BC[:, region_6] / (
        (d4[region_6] - d3[region_6]) + (d5[region_6] - d6[region_6])
    )
    # otherwise it is inside the triangle hence closest point will be the same as the projected position
    distance_to_closest_triangle_point = _batch_norm(
        closest_point_on_triangle_to_projected_point - projected_sphere_center
    )
    no_intersection_idx = np.where(
        (distance_to_closest_triangle_point - triangle_intersection_radius)
        > surface_tol
    )[0]
    return no_intersection_idx, distance_from_triangle_plane


class Grid:
    def __init__(
        self,
        rod: RodType,
        surface: MeshSurface,
        grid_dimension: int = 3,
        exit_boundary_condition: bool = False,
        grid_axes: list = [0, 1],
        **kwargs,
    ):
        super().__init__()
        assert grid_dimension in [2, 3], (
            "Please select a valid grid dimension. Either 2 for 2D grids or 3 for 3D grids"
        )
        self.dimension = grid_dimension
        self.n_faces = surface.n_faces
        self.size = self.compute_grid_size(rod, surface)
        self.min_point = surface.min_point
        self.surface_scale = surface.scale
        self.exit_boundary_condition = exit_boundary_condition
        assert len(grid_axes) == 2, "grid axes must have length 2 for 2D grids"
        for axis in grid_axes:
            assert axis in [0, 1, 2], (
                "grid axes must be one of 0,1,2 for x,y,z axes respectively"
            )
        self.axes = grid_axes
        self._generate_grid(surface)
        if self.dimension == 2:
            self.find_faces = partial(
                self._find_faces_from_2D_grid,
                surface_grid=self.surface_grid,
                x_min=self.min_point[self.axes[0]],
                y_min=self.min_point[self.axes[1]],
                grid_axes=self.axes,
                grid_size=self.size,
                exit_boundary_condition=self.exit_boundary_condition,
            )
        else:
            self.find_faces = partial(
                self._find_faces_from_3D_grid,
                surface_grid=self.surface_grid,
                x_min=self.min_point[0],
                y_min=self.min_point[1],
                z_min=self.min_point[2],
                grid_size=self.size,
                exit_boundary_condition=self.exit_boundary_condition,
            )

    @staticmethod
    def compute_grid_size(rod, surface):
        rod_element_max_dimenension = np.sqrt(
            (2 * max(rod.radius)) ** 2 + (max(rod.rest_lengths)) ** 2
        )
        surface_min_side_length = min(
            np.min(_batch_norm(surface.side_vectors[:, 0, :])),
            np.min(_batch_norm(surface.side_vectors[:, 1, :])),
            np.min(_batch_norm(surface.side_vectors[:, 2, :])),
        )
        return max(rod_element_max_dimenension, surface_min_side_length)

    def _generate_grid(self, surface):
        if self.dimension == 2:
            face_x_left = np.min(surface.faces[self.axes[0], :, :], axis=0)
            face_x_right = np.max(surface.faces[self.axes[0], :, :], axis=0)
            face_y_down = np.min(surface.faces[self.axes[1], :, :], axis=0)
            face_y_up = np.max(surface.faces[self.axes[1], :, :], axis=0)
            self.surface_grid = dict(
                self._create_surface_grid_2D(
                    surface.faces,
                    self.size,
                    self.min_point[self.axes[0]],
                    self.min_point[self.axes[1]],
                    self.axes,
                    face_x_left,
                    face_x_right,
                    face_y_down,
                    face_y_up,
                )
            )
        elif self.dimension == 3:
            face_x_left = np.min(surface.faces[0, :, :], axis=0)
            face_x_right = np.max(surface.faces[0, :, :], axis=0)
            face_y_down = np.min(surface.faces[1, :, :], axis=0)
            face_y_up = np.max(surface.faces[1, :, :], axis=0)
            face_z_back = np.min(surface.faces[2, :, :], axis=0)
            face_z_front = np.max(surface.faces[2, :, :], axis=0)
            self.surface_grid = dict(
                self._create_surface_grid_3D(
                    surface.faces,
                    self.size,
                    self.min_point[0],
                    self.min_point[1],
                    self.min_point[2],
                    face_x_left,
                    face_x_right,
                    face_y_down,
                    face_y_up,
                    face_z_back,
                    face_z_front,
                )
            )

    @staticmethod
    @njit(cache=True)
    def _create_surface_grid_2D(
        faces,
        grid_size,
        x_min,
        y_min,
        grid_axes,
        face_x_left,
        face_x_right,
        face_y_down,
        face_y_up,
    ):
        n_x_positions = int(
            np.ceil((np.max(faces[grid_axes[0], :, :]) - x_min) / grid_size)
        )  # number of grid sizes that fit in x direction
        n_y_positions = int(
            np.ceil((np.max(faces[grid_axes[1], :, :]) - y_min) / grid_size)
        )  # number of grid sizes that fit in y direction
        faces_grid = dict()
        for i in range(n_x_positions):
            x_left = x_min + (
                max(0, i - 1) * grid_size
            )  # current grid square left x coordinate
            x_right = x_min + (
                min(n_x_positions - 1, i + 1) * grid_size
            )  # current grid square right x coordinate
            for j in range(n_y_positions):
                y_down = y_min + (
                    max(0, j - 1) * grid_size
                )  # current grid square down y coordinate
                y_up = y_min + (
                    min(n_y_positions - 1, j + 1) * grid_size
                )  # current grid square up y coordinate
                if (
                    len(
                        np.where(
                            (
                                (
                                    face_y_down > y_up
                                )  # if face_y_down coordinate is greater than grid square up position then face is above grid square
                                + (
                                    face_y_up < y_down
                                )  # if face_y_up coordinate is lower than grid square down position then face is below grid square
                                + (
                                    face_x_right < x_left
                                )  # if face_x_right coordinate is lower than grid square left position then face is to the left of the grid square
                                + (
                                    face_x_left > x_right
                                )  # if face_x_left coordinate is greater than grid square right position then face is to the right of the grid square
                            )
                            == 0  # if the face is not below, above, to the right of or, to the left of the grid then they must intersect
                        )[0]
                    )
                    > 0
                ):
                    faces_grid[(i, j)] = np.where(
                        (
                            (face_y_down > y_up)
                            + (face_y_up < y_down)
                            + (face_x_right < x_left)
                            + (face_x_left > x_right)
                        )
                        == 0
                    )[0]
        return faces_grid

    @staticmethod
    @njit(cache=True)
    def _create_surface_grid_3D(
        faces,
        grid_size,
        x_min,
        y_min,
        z_min,
        face_x_left,
        face_x_right,
        face_y_down,
        face_y_up,
        face_z_back,
        face_z_front,
    ):
        n_x_positions = int(
            np.ceil((np.max(faces[0, :, :]) - x_min) / grid_size)
        )  # number of grid sizes that fit in x direction
        n_y_positions = int(
            np.ceil((np.max(faces[1, :, :]) - y_min) / grid_size)
        )  # number of grid sizes that fit in y direction
        n_z_positions = int(
            np.ceil((np.max(faces[2, :, :]) - z_min) / grid_size)
        )  # number of grid sizes that fit in z direction
        faces_grid = dict()
        for i in range(n_x_positions):
            x_left = x_min + (
                max(0, i - 1) * grid_size
            )  # current grid cube left x coordinate
            x_right = x_min + (
                min(n_x_positions - 1, i + 1) * grid_size
            )  # current grid cube right x coordinate
            for j in range(n_y_positions):
                y_down = y_min + (
                    max(0, j - 1) * grid_size
                )  # current grid cube down y coordinate
                y_up = y_min + (
                    min(n_y_positions - 1, j + 1) * grid_size
                )  # current grid cube up y coordinate
                for k in range(n_z_positions):
                    z_back = z_min + (
                        max(0, k - 1) * grid_size
                    )  # current grid cube back z coordinate
                    z_front = z_min + (
                        min(n_z_positions - 1, k + 1) * grid_size
                    )  # current grid cube front z coordinate
                    if (
                        len(
                            np.where(
                                (
                                    (
                                        face_y_down > y_up
                                    )  # if face_y_down coordinate is greater than grid cube up position then face is above grid cube
                                    + (
                                        face_y_up < y_down
                                    )  # if face_y_up coordinate is lower than grid cube down position then face is below grid cube
                                    + (
                                        face_x_right < x_left
                                    )  # if face_x_right coordinate is lower than grid cube left position then face is to the left of the grid cube
                                    + (
                                        face_x_left > x_right
                                    )  # if face_x_left coordinate is greater than grid cube right position then face is to the right of the grid cube
                                    + (
                                        face_z_front < z_back
                                    )  # if face_z_front coordinate is lower than grid cube back position then face is behind the grid cube
                                    + (
                                        face_z_back > z_front
                                    )  # if face_z_back coordinate is greater than grid cube front position then face is in front of the grid cube
                                )
                                == 0  # if the face is not below, above, behind, in front of, to the right of, or to the left of the grid cube then they must intersect
                            )[0]
                        )
                        > 0
                    ):
                        faces_grid[(i, j, k)] = np.where(
                            (
                                (face_y_down > y_up)
                                + (face_y_up < y_down)
                                + (face_x_right < x_left)
                                + (face_x_left > x_right)
                                + (face_z_front < z_back)
                                + (face_z_back > z_front)
                            )
                            == 0
                        )[0]
        return faces_grid

    @staticmethod
    def _find_faces_from_2D_grid(
        surface_grid,
        x_min,
        y_min,
        grid_axes,
        grid_size,
        position_collection,
        exit_boundary_condition,
    ):
        element_position = _node_to_element_position(position_collection)
        n_element = element_position.shape[-1]
        position_idx_chunks: list[np.ndarray] = []
        face_idx_chunks: list[np.ndarray] = []
        grid_position = np.empty((2, n_element))
        grid_position[0, :] = np.round(
            (element_position[grid_axes[0], :] - x_min) / grid_size
        )
        grid_position[1, :] = np.round(
            (element_position[grid_axes[1], :] - y_min) / grid_size
        )
        # here we take the element position subtract the grid left most lower corner position (to get distance from that point)
        # The distance divided by the grid size converts the distance to units of grid size.
        # Rounding gives us the nearest corner to that element center position
        # Since any grid square can contain at most one element any element will have to lie within the four squares around the corner we found above.

        # find face neighborhood of each element position
        for i in range(n_element):
            try:
                face_idx_element_no_duplicates = surface_grid[
                    (int(grid_position[0, i]), int(grid_position[1, i]))
                ]
            except KeyError:
                face_idx_element_no_duplicates = np.empty((0))
                if (exit_boundary_condition) and (
                    face_idx_element_no_duplicates.size == 0
                ):
                    raise BoundaryError(
                        "Rod outside surface grid boundary"
                    )  # a rod element is within four grid squares with no faces

            n_contacts = face_idx_element_no_duplicates.shape[0]
            if n_contacts > 0:
                face_idx_chunks.append(face_idx_element_no_duplicates)
                position_idx_chunks.append(np.full((n_contacts,), i, dtype=np.int64))

        if len(face_idx_chunks) == 0:
            position_idx_array = np.empty((0,), dtype=np.int64)
            face_idx_array = np.empty((0,), dtype=np.int64)
        else:
            position_idx_array = np.concatenate(position_idx_chunks).astype(
                np.int64, copy=False
            )
            face_idx_array = np.concatenate(face_idx_chunks).astype(
                np.int64, copy=False
            )
        return position_idx_array, face_idx_array, element_position

    @staticmethod
    def _find_faces_from_3D_grid(
        surface_grid,
        x_min,
        y_min,
        z_min,
        grid_size,
        position_collection,
        exit_boundary_condition,
    ):
        """
        here we take the element position subtract the grid left most lower corner position (to get distance from that point)
        The distance divided by the grid size converts the distance to units of grid size.
        Rounding gives us the nearest corner to that element center position
        Since any grid square can contain at most one element any element will have to lie within the 8 cubes around the corner we found above.
        """
        element_position = _node_to_element_position(position_collection)
        n_element = element_position.shape[-1]
        position_idx_chunks: list[np.ndarray] = []
        face_idx_chunks: list[np.ndarray] = []
        grid_position = np.empty((3, n_element))
        grid_position[0, :] = np.round((element_position[0, :] - x_min) / grid_size)
        grid_position[1, :] = np.round((element_position[1, :] - y_min) / grid_size)
        grid_position[2, :] = np.round((element_position[2, :] - z_min) / grid_size)

        # find face neighborhood of each element position
        for i in range(n_element):
            try:
                face_idx_element_no_duplicates = surface_grid[
                    (
                        int(grid_position[0, i]),
                        int(grid_position[1, i]),
                        int(grid_position[2, i]),
                    )
                ]
            except KeyError:
                face_idx_element_no_duplicates = np.empty((0))
                if (exit_boundary_condition) and (
                    face_idx_element_no_duplicates.size == 0
                ):
                    raise BoundaryError(
                        "Rod outside surface grid boundary"
                    )  # a rod element is within eight grid cubes with no faces

            n_contacts = face_idx_element_no_duplicates.shape[0]
            if n_contacts > 0:
                face_idx_chunks.append(face_idx_element_no_duplicates)
                position_idx_chunks.append(np.full((n_contacts,), i, dtype=np.int64))

        if len(face_idx_chunks) == 0:
            position_idx_array = np.empty((0,), dtype=np.int64)
            face_idx_array = np.empty((0,), dtype=np.int64)
        else:
            position_idx_array = np.concatenate(position_idx_chunks).astype(
                np.int64, copy=False
            )
            face_idx_array = np.concatenate(face_idx_chunks).astype(
                np.int64, copy=False
            )
        return position_idx_array, face_idx_array, element_position

    # def save_to(self,name:str,folder="/"):
    #     assert folder[-1]=="/","Folder path must end with /"
    #     if folder=="/":
    #         folder = ""
    #     with open("{0}{1}.dat".format(folder,name), "wb") as file:
    #         dill.dump(self, file)

    # @staticmethod
    # def load(file_path: str):
    #     with open(file_path, "rb") as file:
    #         loaded_object = dill.load(file)
    #         assert isinstance(loaded_object, Grid), (
    #             "Loaded file must be a saved Grid object"
    #         )
    #     return loaded_object
