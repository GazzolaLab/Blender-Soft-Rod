__doc__ = """
BezierSplinePipe: 3D pipe creation using Bezier splines with configurable handle types.

This module provides a robust implementation for creating smooth pipe-like objects
in Blender, with specific attention to preventing abnormal tip appearance that can
occur with improper handle type configurations.
"""
__all__ = ["BezierSplinePipe"]

from typing import TYPE_CHECKING, cast, Any

import warnings
from numbers import Number

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.geometry.protocol import BlenderMeshInterfaceProtocol, SplineDataType
from bsr.tools.keyframe_mixin import KeyFrameControlMixin

from .utils import _validate_position, _validate_radii


class BezierSplinePipe(KeyFrameControlMixin):
    """
    Creates a 3D pipe using Bezier spline curves with proper bevel geometry.

    This class creates smooth pipe-like objects by using Bezier splines with
    configurable handle types to avoid abnormal tip appearance common with
    VECTOR handles.

    Parameters
    ----------
    positions : NDArray
        The position of the spline control points. Shape: (3, n)
    radii : NDArray
        The radius at each control point. Shape: (n,)
    handle_type : str, optional
        Handle type for Bezier points. Options: "AUTO", "VECTOR", "SMOOTH"
        Default: "AUTO" (recommended for smooth pipes without tip distortion)

    Notes
    -----
    - "AUTO" handles provide the smoothest pipe appearance
    - "VECTOR" handles can cause tip distortion and bevel compression at corners
    - "SMOOTH" handles provide manual control but require more setup
    """

    input_states = {"positions", "radii"}
    name = "bspline"

    def __init__(
        self,
        positions: NDArray,
        radii: NDArray,
        handle_type: str = "AUTO",
        downsample_num_element: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Spline constructor

        Parameters
        ----------
        positions : NDArray
            The position of the spline object. (3, n)
        radii : NDArray
            The radius of the spline object. (n,)
        handle_type : str, optional
            The handle type for Bezier points. Options: "AUTO", "VECTOR", "SMOOTH"
            Default is "AUTO" for smoother pipe appearance.
        downsample_num_element : int | None, optional
            If provided, downsample the positions and radii to this number of elements.
            Default is None (no downsampling).
        """
        assert (
            downsample_num_element is None or downsample_num_element >= 2
        ), "downsample_num_element must be at least 2 to include both endpoints"
        self.downsample_num_element = downsample_num_element
        self._obj = self._create_bezier_spline(radii.size, handle_type)
        self._obj.name = self.name
        self.update_states(positions, radii)

        self._material = bpy.data.materials.new(
            name=f"{self._obj.name}_material"
        )
        self._obj.data.materials.append(self._material)

    @classmethod
    def create(
        cls,
        states: SplineDataType,
        handle_type: str = "AUTO",
        downsample_num_element: int | None = None,
    ) -> "BezierSplinePipe":
        """
        Basic factory method to create a new spline object.

        Parameters
        ----------
        states : SplineDataType
            Dictionary containing 'positions', 'radii'
        handle_type : str, optional
            The handle type for Bezier points. Options: "AUTO", "VECTOR", "SMOOTH"
            Default is "AUTO" for smoother pipe appearance.
        downsample_num_element : int | None, optional
            If provided, downsample the positions and radii to this number of elements.
            Default is None (no downsampling).
        """

        # TODO: Refactor this part: never copy-paste code. Make separate function in utils.py
        remaining_keys = set(states.keys()) - cls.input_states
        if len(remaining_keys) > 0:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        return cls(
            states["positions"],
            states["radii"],
            handle_type,
            downsample_num_element,
        )

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

    def update_states(
        self, positions: NDArray | None = None, radii: NDArray | None = None
    ) -> None:
        """
        Updates the position and radius of the spline object.

        Parameters
        ----------
        positioni : NDArray
            The new position of the spline object.
        radii : float
            The new radius of the spline object.

        Raises
        ------
        ValueError
            If the shape of the position or radius is incorrect, or if the data is NaN.
        """

        # Apply downsampling if configured
        if self.downsample_num_element is not None:
            positions, radii = self._downsample_data(positions, radii)

        spline = self.object.data.splines[0]
        if positions is not None:
            _validate_position(positions)
            for i, point in enumerate(spline.bezier_points):
                x, y, z = positions[:, i]
                point.co = (x, y, z)
        if radii is not None:
            _validate_radii(radii)
            for i, point in enumerate(spline.bezier_points):
                point.radius = radii[i]

    def _downsample_data(
        self, positions: NDArray, radii: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Downsamples the positions and radii arrays to the specified number of elements.
        Always includes both endpoints (first and last points).

        Parameters
        ----------
        positions : NDArray
            The position array to downsample. Shape: (3, n)
        radii : NDArray
            The radius array to downsample. Shape: (n,)
        num_elements : int
            The target number of elements after downsampling.

        Returns
        -------
        tuple[NDArray, NDArray]
            Downsampled positions and radii arrays.
        """
        num_elements = self.downsample_num_element
        if positions.shape[1] <= num_elements:
            # No downsampling needed
            # TODO: should we just raise error, or interpolate the positions and radii?
            return positions, radii

        t = np.linspace(0, 1, num_elements)
        t_old = np.linspace(0, 1, positions.shape[1])
        positions[0, :] = np.interp(t, t_old, positions[0, :])
        positions[1, :] = np.interp(t, t_old, positions[1, :])
        positions[2, :] = np.interp(t, t_old, positions[2, :])
        radii = np.interp(t, t_old, radii)

        return positions, radii

    def _create_bezier_spline(
        self, number_of_points: int, handle_type: str = "AUTO"
    ) -> bpy.types.Object:
        """
        Creates a new pipe object.

        Parameters
        ----------
        number_of_points : int
            The number of points in the pipe.
        handle_type : str, optional
            The handle type for Bezier points. Options: "AUTO", "VECTOR", "SMOOTH"
            Default is "AUTO" for smoother pipe appearance.
        """
        # Create a new curve
        curve_data = bpy.data.curves.new(name="spline_curve", type="CURVE")
        curve_data.dimensions = "3D"

        spline = curve_data.splines.new(type="BEZIER")
        spline.bezier_points.add(
            number_of_points - 1
        )  # First point is already there

        # Set the spline points and radii
        for i in range(number_of_points):
            point = spline.bezier_points[i]
            point.handle_left_type = point.handle_right_type = handle_type

        # Create a new object with the curve data
        curve_object = bpy.data.objects.new("spline_curve_object", curve_data)
        curve_object.data.resolution_u = 1
        curve_object.data.render_resolution_u = 1
        bpy.context.collection.objects.link(curve_object)

        # Create a bevel object for the pipe profile
        bpy.ops.curve.primitive_bezier_circle_add(
            radius=1,
            enter_editmode=False,
            align="WORLD",
            location=(0, 0, 0),
            scale=(1, 1, 1),
        )
        bevel_circle = bpy.context.object
        bevel_circle.name = "bevel_circle"
        # Set resolution for smoother bevel profile
        bevel_circle.data.resolution_u = 1
        # Hide the bevel circle object in the viewport and render
        bevel_circle.hide_viewport = True
        bevel_circle.hide_render = True

        # Set the bevel object to the curve
        curve_data.bevel_mode = "OBJECT"
        curve_data.bevel_object = bevel_circle
        curve_data.use_fill_caps = True

        return curve_object

    def update_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        spline = self.object.data.splines[0]
        for i, point in enumerate(spline.bezier_points):
            point.keyframe_insert(data_path="co", frame=keyframe)
            point.keyframe_insert(data_path="radius", frame=keyframe)
        self.material.keyframe_insert(data_path="diffuse_color", frame=keyframe)


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]).T,
        "radii": np.array([1.0, 1.0]),
    }
    _: BlenderMeshInterfaceProtocol = BezierSplinePipe.create(data)
