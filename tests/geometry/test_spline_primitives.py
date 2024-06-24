import pytest
import numpy as np

from bsr.geometry.primitive.pipe import BezierSplinePipe


@pytest.mark.parametrize(
    "possible_bezier_spline_data",
    [
        # TODO:
        dict(),  # Empty dictionary
        dict(positions=..., radii=...),  # Change both positions and radii
        dict(radii=...),  # Change only radii
        dict(positions=...),  # Change only positions
    ],
)
def test_update_states_with_data(possible_bezier_spline_data):
    default_data = dict(
        # TODO: Default pipe shape
    )
    primitive = BezierSplinePipe.create(default_data)
    assert primitive.object is not None

    # TODO: check if create primitive has corresponding "default_data" geometry
    ...

    # Run update_states without error
    primitive.update_states(**possible_bezier_spline_data)
    default_data.update(possible_bezier_spline_data)  # Update state dictionary

    # TODO: check if create primitive has updated geometry
    ...


@pytest.mark.parametrize(
    "impossible_shaped_data",
    [
        # TODO: Test with impossible shape. Make sure to test impossible radius (negative or zero), and different shape)
    ],
)
def test_update_states_with_wrong_shape(impossible_shaped_data):
    default_data = dict(
        # TODO: Default pipe shape
    )
    primitive = BezierSplinePipe.create(default_data)
    with pytest.raises(ValueError):
        primitive.update_states(**impossible_shaped_data)


@pytest.mark.parametrize(
    "nan_data",
    [
        # TODO
        dict(), # nan in position data
        dict(), # nan in radius data
    ],
)
def test_update_states_with_nan_values(nan_data):
    default_data = dict(
        # TODO: Default pipe shape
    )
    primitive = BezierSplinePipe.create(default_data)
    with pytest.raises(ValueError) as exc_info:
        primitive.update_states(**nan_data)
    assert "contains NaN" in str(exc_info.value)


def test_bezier_spline_creator():
    default_data = dict(
        # TODO: Default pipe shape
    )
    primitive = BezierSplinePipe.create(default_data)
    old_bezier_spline = primitive.object
    new_bezier_spline = primitive._create_bezier_spline()
    assert new_bezier_spline is not None
    assert old_bezier_spline is not new_bezier_spline
    assert isinstance(new_bezier_spline, bpy_types.Object)

