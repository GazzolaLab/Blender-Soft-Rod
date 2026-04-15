__doc__ = """ surface classes and implementation details """

import numpy as np
from numpy.testing import assert_allclose
import pyvista as pv

from elastica.surface.plane import Plane
from elastica.utils import Tolerance


class MeshSurface(Plane):
    """
    Mesh surface object. Initalized using PyVista mesh.
    Attributes
    ----------
    mesh: pyvista PolyData object.
        Surface mesh.
    """

    def __init__(
        self,
        mesh: pv.PolyData,
    ):
        self.mesh = mesh
        self._mesh_surface_init()

    def _mesh_surface_init(self):
        self.scale = self.scale_calculation(self.mesh.bounds)
        self.faces = self.face_calculation(
            self.mesh.faces, self.mesh.points, self.mesh.n_faces_strict
        )
        self.face_indices_calculation(
            self.mesh.faces,
            self.mesh.n_faces_strict,
        )
        self.vertices = self.mesh.points.T
        try:
            self.vertex_normals = self.mesh.active_normals.T
        except AttributeError:
            self.vertex_normals = None
        try:
            self.texture_vertices = self.mesh.active_texture_coordinates.T
        except AttributeError:
            self.texture_vertices = None
        self.face_centers = self.face_center_calculation(self.faces, self.n_faces)
        self.face_normals = self.face_normal_calculation(
            self.mesh.faces,
            self.mesh.face_normals,
            self.n_faces,
            self.mesh.n_faces_strict,
        )
        for face in range(self.n_faces):
            assert_allclose(
                np.linalg.norm(self.face_normals[:, face]),
                1.0,
                atol=Tolerance.atol(),
                err_msg="face {0}'s normal is not a unit vector".format(face),
            )

        self.min_point = np.array(
            [
                np.min(self.faces[0, :, :]),
                np.min(self.faces[1, :, :]),
                np.min(self.faces[2, :, :]),
            ]
        )  # grid x zero position
        self.side_vectors = np.zeros((3, 3, self.n_faces))  # coords,sides,faces
        self.side_vectors[:, 0, :] = self.faces[:, 1, :] - self.faces[:, 0, :]  # AB
        self.side_vectors[:, 1, :] = self.faces[:, 2, :] - self.faces[:, 0, :]  # AC
        self.side_vectors[:, 2, :] = self.faces[:, 2, :] - self.faces[:, 1, :]  # BC

    @staticmethod
    def scale_calculation(bounds: np.ndarray) -> np.ndarray:
        """
        This function calculates scale of the mesh,
        for that it calculates the maximum distance between mesh's farthest verticies in each axis.
        """
        scale = np.zeros(3)
        axis = 0
        for i in range(0, 5, 2):
            scale[axis] = bounds[i + 1] - bounds[i]
            axis += 1

        return scale

    @staticmethod
    def face_calculation(
        pvfaces: np.ndarray, meshpoints: np.ndarray, n_pv_faces: int
    ) -> np.ndarray:
        """
        This function converts the faces from pyvista to pyelastica geometry

        What the function does?:
        ------------------------

        # The pyvista's 'faces' attribute returns the connectivity array of the faces of the mesh.
            ex: [3, 0, 1, 2, 4, 0, 1, 3, 4]
            The faces array is organized as:
                [n0, p0_0, p0_1, ..., p0_n, n1, p1_0, p1_1, ..., p1_n, ...]
                    ,where n0 is the number of points in face 0, and pX_Y is the Y'th point in face X.
            For more info, refer to the api reference here - https://docs.pyvista.org/version/stable/api/core/_autosummary/pyvista.PolyData.faces.html

        # The pyvista's 'points' attribute returns the individual vertices of the mesh with no connection information.
            ex: [-1.  1. -1.]
                [ 1. -1. -1.]
                [ 1.  1. -1.]

        # This function takes the 'mesh.points' and numbers them as 0, 1, 2 ..., n_faces - 1;
          then establishes connection between verticies of same cell/face through the 'mesh.faces' array
          and returns an array with dimension (3 spatial coordinates, 3 vertices, n faces), where n_faces is the number of faces in the mesh.

        Notes:
        ------

        - This function works only if each face of the mesh has equal no. of vertices i.e
          all the faces of the mesh has similar geometry.
        """
        vertex_no = 0
        n_faces = 0
        faces_temp = []
        for i in range(n_pv_faces):
            if pvfaces[vertex_no] == 3:
                vertex_no += 1
                # this current face is a triangle
                n_faces += 1
                face = []
                for j in range(3):
                    face.append(meshpoints[pvfaces[vertex_no + j]])
                vertex_no += 3
                faces_temp.append(face)
            else:
                vertex_no += 1
                # this current face has 4 vertices so we subdivide into two triangles
                n_faces += 2
                face_1 = []
                face_2 = []
                # first triangle
                face_1.append(meshpoints[pvfaces[vertex_no]])
                face_1.append(meshpoints[pvfaces[vertex_no + 1]])
                face_1.append(meshpoints[pvfaces[vertex_no + 2]])

                # second triangle
                face_2.append(meshpoints[pvfaces[vertex_no]])
                face_2.append(meshpoints[pvfaces[vertex_no + 2]])
                face_2.append(meshpoints[pvfaces[vertex_no + 3]])

                faces_temp.append(face_1)
                faces_temp.append(face_2)
                vertex_no += 4

        faces = np.zeros((3, 3, n_faces))
        for i in range(n_faces):
            for j in range(3):
                faces[..., j, i] = faces_temp[i][j]

        return faces

    def face_indices_calculation(self, pvfaces: np.ndarray, n_pv_faces: int):
        face_indices = []
        vertex_no = 0
        face_count = 0
        for i in range(n_pv_faces):
            if pvfaces[vertex_no] == 3:
                # face is a triangle
                face_indices.append(
                    pvfaces[[vertex_no + 1, vertex_no + 2, vertex_no + 3]]
                )
                vertex_no += 4
                face_count += 1
            else:
                # face has 4 vertices
                face_indices.append(
                    pvfaces[[vertex_no + 1, vertex_no + 2, vertex_no + 3]]
                )
                face_indices.append(
                    pvfaces[[vertex_no + 1, vertex_no + 3, vertex_no + 4]]
                )
                vertex_no += 5
                face_count += 2
        self.n_faces = face_count
        self.face_indices = np.array(face_indices).T

    @staticmethod
    def face_normal_calculation(
        pvfaces: np.ndarray,
        pyvista_face_normals: np.ndarray,
        n_faces: int,
        n_pv_faces: int,
    ) -> np.ndarray:
        """
        This function converts the face normals from pyvista to pyelastica geometry,
        in pyelastica the face are stored in the format of (n_pv_faces, 3 spatial coordinates),
        this is converted into (3 spatial coordinates, n_faces).
        """
        vertex_no = 0
        face_count = 0
        face_normals = np.zeros((3, n_faces))
        for i in range(n_pv_faces):
            if pvfaces[vertex_no] == 3:
                # face is a triangle
                vertex_no += 4
                face_normals[:, face_count] = pyvista_face_normals[i, :]
                face_count += 1
            else:
                # face has 4 vertices, both triangles share same normal
                vertex_no += 5
                face_normals[:, face_count] = pyvista_face_normals[i, :]
                face_normals[:, face_count + 1] = pyvista_face_normals[i, :]
                face_count += 2

        return face_normals

    @staticmethod
    def face_center_calculation(faces: np.ndarray, n_faces: int) -> np.ndarray:
        """
        This function calculates the position vector of each face of the mesh
        simply by averaging all the vertices of every face/cell.
        """
        face_centers = np.zeros((3, n_faces))

        for i in range(n_faces):
            for j in range(3):
                temp_sum = faces[j][..., i].sum()
                face_centers[j][i] = temp_sum / 3

        return face_centers
