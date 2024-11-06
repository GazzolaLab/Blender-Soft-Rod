__doc__ = """
This module provides a set of geometry-mesh interfaces for blender objects.
"""
__all__ = ["Sphere", "Cylinder"]

from typing import TYPE_CHECKING, Any, cast

import warnings
from numbers import Number

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.geometry.protocol import BlenderMeshInterfaceProtocol, MeshDataType
from bsr.tools.keyframe_mixin import KeyFrameControlMixin


def calculate_cylinder_orientation(
    position_1: NDArray, position_2: NDArray
) -> tuple[float, NDArray, NDArray]:
    """
    Calculates the centerpoint, depth, and rotational angle of the cylinder object.

    Parameters
    ----------
    position_1 : NDArray
        One endpoint position of the cylinder object. Expected shape is (n_dim,)
        n_dim = 3
    position_2: NDArray
        Other endpoint position of the cylinder object. Expected shape is (n_dim,)
        n_dim = 3

    Returns
    -------
    tuple: float, NDArray, NDArray
        Tuple containing the values for the depth, centerpoint and rotation angle.
        Expected shape is (n_dim, n_dim)
        n_dim = 3
    """

    depth = np.linalg.norm(position_2 - position_1)
    dz = position_2[2] - position_1[2]
    dy = position_2[1] - position_1[1]
    dx = position_2[0] - position_1[0]
    center = (position_1 + position_2) / 2.0
    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz / depth)
    angles = np.array([phi, theta])
    return float(depth), center, angles


def _validate_position(position: NDArray) -> None:
    """
    Checks if inputted position values are valid

    Parameters
    ---------
    position: NDArray
        Position input (endpoint or centerpoint depending on Object type)

    Raises
    ------
    ValueError
        If the position is the wrong shape or contains NaN values
    """

    if position.shape != (3,):
        raise ValueError("The shape of the position is incorrect.")
    if np.isnan(position).any():
        raise ValueError("The position contains NaN values.")


def _validate_radius(radius: float) -> None:
    """
    Checks if inputted radius value is valid

    Parameters:
    -----------
    radius: Float
        Radius input

    Raises
    ------
    ValueError
        If the radius is not positive, or contains NaN values
    """

    if not isinstance(radius, Number) or radius <= 0:
        raise ValueError("The radius must be a positive float.")
    if np.isnan(radius):
        raise ValueError("The radius contains NaN values.")


class Sphere(KeyFrameControlMixin):
    """
    This class provides a mesh interface for Blender Sphere objects.
    Sphere objects are created with the given position and radius.

    Parameters
    ----------
    position : NDArray
        The position of the sphere object. Expected shape is (n_dim,)
        n_dim = 3
    radius : float
        The radius of the sphere object.
    """

    input_states = {"position", "radius"}

    def __init__(self, position: NDArray, radius: float) -> None:
        """
        Sphere class constructor
        """
        self._obj = self._create_sphere()
        self._material = bpy.data.materials.new(
            name=f"{self._obj.name}_material"
        )
        self._obj.data.materials.append(self._material)
        self.update_states(position, radius)

    # TODO: Find better way to represnet radius
    @classmethod
    def create(cls, states: MeshDataType) -> "Sphere":
        """
        Basic factory method to create a new Sphere object.
        States must have the following keys: position(n_dim,), radius(float)


        Parameters
        ----------
        states: MeshDataType
            A dictionary containing the sphere's center position and radius

        Returns
        -------
        Sphere
            A sphere object with the defined center position and radius

        Raises
        ------
        Warning
        If unused keys are present in the dictionary within states
        """

        remaining_keys = set(states.keys()) - cls.input_states
        if len(remaining_keys) > 0:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        return cls(states["position"], states["radius"])

    @property
    def material(self) -> bpy.types.Material:
        """
        Access the Blender material.
        """

        return self._material

    @property
    def object(self) -> bpy.types.Object:
        """
        Access the Blender object.
        """

        return self._obj

    def update_states(
        self, position: NDArray | None = None, radius: float | None = None
    ) -> None:
        """
        Updates the position and radius of the sphere object.

        Parameters
        ----------
        position : NDArray
            The new position of the sphere object.
        radius : float
            The new radius of the sphere object.

        Raises
        ------
        ValueError
            If the shape of the position or radius is incorrect, or if the data is NaN.
        """

        if position is not None:
            _validate_position(position)
            self.object.location.x = position[0]
            self.object.location.y = position[1]
            self.object.location.z = position[2]
        if radius is not None:
            _validate_radius(radius)
            self.object.scale = (radius, radius, radius)

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the sphere object.

        Parameters
        ----------
        color : NDArray
            The new color of the sphere object in RGBA format.
        """

        if "color" in kwargs:
            color = kwargs["color"]
            if isinstance(color, (tuple, list)):
                color = np.array(color)
            assert isinstance(
                color, np.ndarray
            ), "Keyword argument `color` should be a numpy array."
            assert color.shape == (
                4,
            ), "Keyword argument color should be a 1D array with 4 elements: RGBA."
            assert np.all(color >= 0) and np.all(
                color <= 1
            ), "Keyword argument color should be in the range of [0, 1]."
            self.material.diffuse_color = tuple(color)

    def _create_sphere(self) -> bpy.types.Object:
        """
        Creates a new sphere object with the given position and radius.
        """
        bpy.ops.mesh.primitive_uv_sphere_add()
        return bpy.context.active_object

    def update_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self.object.keyframe_insert(data_path="location", frame=keyframe)
        self.material.keyframe_insert(data_path="diffuse_color", frame=keyframe)


class Cylinder(KeyFrameControlMixin):
    """
    This class provides a mesh interface for Blender Cylinder objects.
    Cylinder objects are created with the given endpoint positions and radius.

    Parameters
    ----------
    position_1 : NDArray
        The first endpoint position of the cylinder object. (3D)
    position_2 : NDArray
        The second endpoint position of the cylinder object. (3D)
    radius : float
        The radius of the cylinder object.
    """

    input_keys = {"position_1", "position_2", "radius"}

    def __init__(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius: float,
    ) -> None:
        """
        Cylinder class constructor
        """
        self._obj = self._create_cylinder()
        self._material = bpy.data.materials.new(
            name=f"{self._obj.name}_material"
        )
        self._obj.data.materials.append(self._material)
        # FIXME: This is a temporary solution
        # Ideally, these modules should not contain any data
        self._states_position_1 = position_1
        self._states_position_2 = position_2
        self._states_radius = radius
        self.update_states(position_1, position_2, radius)

    @classmethod
    def create(cls, states: MeshDataType) -> "Cylinder":
        """
        Basic factory method to create a new Cylinder object.

        Parameters
        ----------
        states: MeshDataType
            A dictionary containing the cylinder's endpoint positions and radius

        Returns
        -------
        Cylinder
            A Cylinder object with the defined endpoint positions and radius

        Raises
        ------
        Warning
        If unused keys are present in the dictionary within states
        """

        remaining_keys = set(states.keys()) - cls.input_keys
        if len(remaining_keys) > 0:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        return cls(states["position_1"], states["position_2"], states["radius"])

    @property
    def material(self) -> bpy.types.Material:
        """
        Access the Blender material.
        """

        return self._material

    @property
    def object(self) -> bpy.types.Object:
        """
        Access the Blender object.
        """

        return self._obj

    def update_states(
        self,
        position_1: NDArray | None = None,
        position_2: NDArray | None = None,
        radius: float | None = None,
    ) -> None:
        """
        Updates the position and radius of the cylinder object.

        Parameters
        ----------
        position_1 : NDArray
            The first new endpoint position of the cylinder object.
        position_2 : NDArray
            The second new endpoint position of the cylinder object.
        radius : float
            The new radius of the cylinder object.

        Raises
        ------
        ValueError
            If the shape of the positions or radius is incorrect, or if the data is NaN.
        """
        if position_1 is None and position_2 is None and radius is None:
            return
        if position_1 is not None:
            position_1 = cast(NDArray[np.floating], position_1)
            _validate_position(position_1)
            self._states_position_1 = position_1
        else:
            position_1 = self._states_position_1
        if position_2 is not None:
            position_2 = cast(NDArray[np.floating], position_2)
            _validate_position(position_2)
            self._states_position_2 = position_2
        else:
            position_2 = self._states_position_2
        if radius is not None:
            _validate_radius(radius)
            self._states_radius = radius
        else:
            radius = self._states_radius

        # Validation check
        if np.allclose(position_1, position_2):
            raise ValueError(
                f"Two positions must be different: {(position_1 - position_2)=}"
            )

        depth, center, angles = calculate_cylinder_orientation(
            position_1, position_2
        )
        self.object.location = center
        self.object.rotation_euler = (0, angles[1], angles[0])
        self.object.scale[2] = depth
        self.object.scale[0] = radius
        self.object.scale[1] = radius

    def update_material(self, **kwargs: dict[str, Any]) -> None:
        """
        Updates the material of the cylinder object.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments for the material update.
        """

        if "color" in kwargs:
            color = kwargs["color"]
            if isinstance(color, (tuple, list)):
                color = np.array(color)
            assert isinstance(
                color, np.ndarray
            ), "Keyword argument `color` should be a numpy array."
            assert color.shape == (
                4,
            ), "Keyword argument `color` should be a 1D array with 4 elements: RGBA."
            assert np.all(color >= 0) and np.all(
                color <= 1
            ), "Values of the keyword argument `color` should be in the range of [0, 1]."
            self.material.diffuse_color = tuple(color)

    def _create_cylinder(
        self,
    ) -> bpy.types.Object:
        """
        Creates a new cylinder object.
        """
        bpy.ops.mesh.primitive_cylinder_add(
            radius=1.0,
            depth=1.0,
        )  # Fix keep these values as default.
        cylinder = bpy.context.active_object
        return cylinder

    def update_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self.object.keyframe_insert(data_path="location", frame=keyframe)
        self.object.keyframe_insert(data_path="rotation_euler", frame=keyframe)
        self.object.keyframe_insert(data_path="scale", frame=keyframe)
        self.material.keyframe_insert(data_path="diffuse_color", frame=keyframe)


# TODO: Will be implemented in the future
class Frustum(KeyFrameControlMixin):  # pragma: no cover
    """
    This class provides a mesh interface for Blender Frustum objects.
    Frustum objects are created with the given positions and radii.

    Parameters
    ----------
    position_1 : NDArray
        The position of the first end of the frustum object. (3D)
    position_2 : NDArray
        The position of the second end of the frustum object. (3D)
    radius_1 : float
        The radius of the first end of the frustum object.
    radius_2 : float
        The radius of the second end of the frustum object.
    """

    input_keys = {"position_1", "position_2", "radius_1", "radius_2"}

    def __init__(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius_1: float,
        radius_2: float,
    ) -> None:
        raise NotImplementedError
        # self._obj = self._create_frustum(
        #    position_1, position_2, radius_1, radius_2
        # )
        # self.update_states(position_1, position_2, radius_1, radius_2)

        # self.mat = bpy.data.materials.new(name="cyl_mat")
        # self.obj.active_material = self.mat

    @classmethod
    def create(cls, states: MeshDataType) -> "Frustum":
        raise NotImplementedError
        # return cls(
        #    states["position_1"],
        #    states["position_2"],
        #    states["radius_1"],
        #    states["radius_2"],
        # )

    @property
    def object(self) -> bpy.types.Object:
        raise NotImplementedError

    def _create_frustum(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius_1: float,
        radius_2: float,
    ) -> bpy.types.Object:
        raise NotImplementedError
        # depth, center, angles = calculate_cylinder_orientation(
        #     position_1, position_2
        # )
        # bpy.ops.mesh.primitive_cone_add(
        #     radius1=radius_1, radius2=radius_2, depth=1,
        # )
        # frustum = bpy.context.active_object
        # frustum.rotation_euler = (0, angles[1], angles[0])
        # frustum.location = center
        # frustum.scale[2] = depth
        # return frustum

    def update_states(
        self,
        position_1: NDArray,
        position_2: NDArray,
        radius_1: float,
        radius_2: float,
    ) -> None:
        raise NotImplementedError

    def update_keyframe(self, keyframe: int) -> None:
        raise NotImplementedError


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {"position": np.array([0, 0, 0]), "radius": 1.0}
    _: BlenderMeshInterfaceProtocol = Sphere.create(data)
    data = {
        "position_1": np.array([0, 0, 0]),
        "position_2": np.array([1, 1, 1]),
        "radius": 1.0,
    }
    _: BlenderMeshInterfaceProtocol = Cylinder.create(data)  # type: ignore[no-redef]
    data = {
        "position_1": np.array([0, 0, 0]),
        "position_2": np.array([1, 1, 1]),
        "radius_1": 1.0,
        "radius_2": 1.5,
    }
    _: BlenderMeshInterfaceProtocol = Frustum.create(data)  # type: ignore[no-redef]
