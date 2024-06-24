# TODO: documentation
__doc__ = """
"""
__all__ = ["BezierSplinePipe"]

from typing import TYPE_CHECKING, cast

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
    TODO: Documentation

    Parameters
    ----------
    positions : NDArray
        The position of the spline object. (3, n)
    radii : float
        The radius of the spline object. (3, n)

    """

    input_states = {"positions", "radii"}

    def __init__(self, positions: NDArray, radii: NDArray) -> None:
        """
        Spline constructor
        """

        self._obj = self._create_bezier_spline(radii.size)
        self.update_states(positions, radii)

    @classmethod
    def create(cls, states: SplineDataType) -> "BezierSplinePipe":
        """
        Basic factory method to create a new spline object.
        """

        # TODO: Refactor this part: never copy-paste code. Make separate function in utils.py
        remaining_keys = set(states.keys()) - cls.input_states
        if len(remaining_keys) > 0:
            warnings.warn(
                f"{list(remaining_keys)} are not used as a part of the state definition."
            )
        return cls(states["position"], states["radius"])

    @property
    def object(self) -> bpy.types.Object:
        """
        Access the Blender object.
        """

        return self._obj

    def update_states(
        self, positions: NDArray | None = None, radii: float | None = None
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

        spline = self.object.splines[0]
        if positions is not None:
            _validate_position(positions)
            for i, point in enumerate(spline.bezier_points):
                x, y, z = positions[:, i]
                point.co = (x, y, z)
        if radii is not None:
            _validate_radii(radii)
            for i, point in enumerate(spline.bezier_points):
                point.radius = radii[i]

    def _create_bezier_spline(self, number_of_points: int) -> bpy.types.Object:
        """
        Creates a new pipe object.

        Parameters
        ----------
        number_of_points : int
            The number of points in the pipe.
        """
        # Create a new curve
        curve_data = bpy.data.curves.new(name="spline_curve", type="CURVE")
        curve_data.dimensions = "3D"

        spline = curve_data.splines.new(type="BEZIER")
        spline.bezier_points.add(number_of_points - 1)

        # Set the spline points and radii
        for i in range(number_of_points):
            point = spline.bezier_points[i]
            point.handle_left_type = point.handle_right_type = "AUTO"

        # Create a new object with the curve data
        curve_object = bpy.data.objects.new("spline_curve_object", curve_data)
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
        # Hide the bevel circle object in the viewport and render
        bevel_circle.hide_viewport = True
        bevel_circle.hide_render = True

        # Set the bevel object to the curve
        curve_data.bevel_mode = "OBJECT"
        curve_data.bevel_object = bevel_circle
        curve_data.use_fill_caps = True

        return curve_data

    def set_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self.object.keyframe_insert(data_path="location", frame=keyframe)
        self.object.keyframe_insert(data_path="radius", frame=keyframe)


if TYPE_CHECKING:
    # This is required for explicit type-checking
    data = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]]),
        "radii": np.array([1.0, 1.0]),
    }
    _: BlenderMeshInterfaceProtocol = BezierSplinePipe.create(data)
