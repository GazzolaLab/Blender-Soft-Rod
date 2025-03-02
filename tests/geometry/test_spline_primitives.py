import bpy.types as bpy_types
import numpy as np
import pytest

from bsr.geometry.primitives.pipe import BezierSplinePipe


@pytest.fixture
def default_data():
    return dict(
        positions=np.array([[0, 0, 0], [1, 1, 1]]).T,
        radii=np.array([1, 1]),
    )


def assert_bezier_positions(spline, positions):
    spline_positions = np.asarray(
        [list(point.co) for point in spline.bezier_points]
    ).T
    np.testing.assert_array_almost_equal(spline_positions, positions)


def assert_bezier_radii(spline, radii):
    spline_radii = np.asarray([point.radius for point in spline.bezier_points])
    np.testing.assert_array_almost_equal(spline_radii, radii)


@pytest.mark.parametrize(
    "possible_bezier_spline_data",
    [
        # TODO:
        dict(),  # Empty dictionary
        dict(
            positions=np.array([[0, 0, 0], [1, 1, 2]]).T, radii=np.array([1, 2])
        ),  # Change both positions and radii
        dict(radii=np.array([1, 2])),  # Change only radii
        dict(
            positions=np.array([[0, 0, 0], [1, 1, 2]]).T
        ),  # Change only positions
    ],
)
def test_update_states_with_data(default_data, possible_bezier_spline_data):
    primitive = BezierSplinePipe.create(default_data)
    assert primitive.object is not None

    # check if create primitive has corresponding "default_data" geometry
    if "positions" in possible_bezier_spline_data:
        assert_bezier_positions(
            primitive.object.splines[0], default_data["positions"]
        )
    if "radii" in possible_bezier_spline_data:
        assert_bezier_radii(primitive.object.splines[0], default_data["radii"])

    # Run update_states without error
    primitive.update_states(**possible_bezier_spline_data)
    default_data.update(possible_bezier_spline_data)  # Update state dictionary

    # check if create primitive has updated geometry
    if "positions" in possible_bezier_spline_data:
        assert_bezier_positions(
            primitive.object.splines[0],
            possible_bezier_spline_data["positions"],
        )
    if "radii" in possible_bezier_spline_data:
        assert_bezier_radii(
            primitive.object.splines[0], possible_bezier_spline_data["radii"]
        )


@pytest.mark.parametrize(
    "impossible_shaped_data",
    [
        # Test with impossible shape. Make sure to test impossible radii (negative or zero), and different shape)
        dict(radii=np.array([1, -1, 3])),
        dict(radii=np.array([0])),
    ],
)
def test_update_states_with_wrong_shape(default_data, impossible_shaped_data):
    primitive = BezierSplinePipe.create(default_data)
    with pytest.raises(ValueError):
        primitive.update_states(**impossible_shaped_data)


@pytest.mark.parametrize(
    "nan_data",
    [
        dict(
            positions=np.array([[np.nan, 0, 0], [1, 1, 1]]).T
        ),  # nan in position data
        dict(radii=np.array([1, np.nan])),  # nan in radii data
    ],
)
def test_update_states_with_nan_values(default_data, nan_data):
    primitive = BezierSplinePipe.create(default_data)
    with pytest.raises(ValueError) as exc_info:
        primitive.update_states(**nan_data)
    assert "contains NaN" in str(exc_info.value)


def test_bezier_spline_creator(
    default_data,
):
    primitive = BezierSplinePipe.create(default_data)
    old_bezier_spline = primitive.object
    new_bezier_spline = primitive._create_bezier_spline(
        len(default_data["radii"])
    )
    assert new_bezier_spline is not None
    assert old_bezier_spline is not new_bezier_spline
    assert isinstance(new_bezier_spline, bpy_types.Curve)
