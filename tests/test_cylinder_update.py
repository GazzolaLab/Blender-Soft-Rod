import bpy.types as bpy_types
import numpy as np
import pytest

from bsr.geometry import Cylinder


@pytest.mark.parametrize(
    "possible_cylinder_data",
    [
        dict(
            position_1=np.array([10, 10, 10]),
            position_2=np.array([20, 20, 20]),
            radius=10.0,
        ),
        dict(position_1=np.array([10, 10, 10]), radius=10.0),
        dict(radius=10.0),
        dict(position_1=np.array([10, 10, 10])),
    ],
)
def test_update_states_with_data(possible_cylinder_data):
    default_data = dict(
        position_1=np.array([0, 0, 0]),
        position_2=np.array([1, 1, 1]),
        radius=1.0,
    )
    primitive = Cylinder.create(default_data)
    assert primitive.object is not None
    np.testing.assert_allclose(
        primitive.object.scale, np.array([1.0, 1.0, np.sqrt(3)])
    )
    np.testing.assert_allclose(
        primitive.object.location, np.array([0.5, 0.5, 0.5])
    )

    primitive.update_states(**possible_cylinder_data)

    default_data.update(possible_cylinder_data)  # Update state dictionary
    if (
        "position_1" in possible_cylinder_data
        or "position_2" in possible_cylinder_data
    ):
        center_point = (
            default_data["position_1"] + default_data["position_2"]
        ) / 2.0
        depth = np.linalg.norm(
            default_data["position_1"] - default_data["position_2"]
        )
        np.testing.assert_allclose(primitive.object.location, center_point)
        np.testing.assert_allclose(primitive.object.scale[2], depth)
    if "radius" in possible_cylinder_data:
        np.testing.assert_allclose(
            primitive.object.scale[0], default_data["radius"]
        )
        np.testing.assert_allclose(
            primitive.object.scale[1], default_data["radius"]
        )


@pytest.mark.parametrize(
    "impossible_shaped_data",
    [
        dict(position_1=np.array([10, 10, 10, 10]), radius=10.0),
        dict(position_2=np.array([10, 10, 10, 10]), radius=10.0),
        dict(position_1=np.array([0, 0, 0]), position_2=np.array([0, 0, 0])),
        dict(radius=np.array([10, 10, 10])),
        dict(radius=-1),
        dict(radius=0),
        dict(position_1=np.array([10, 10])),
    ],
)
def test_update_states_with_wrong_shape(impossible_shaped_data):
    default_data = dict(
        position_1=np.array([0, 0, 0]),
        position_2=np.array([1, 1, 1]),
        radius=1.0,
    )
    primitive = Cylinder.create(default_data)
    with pytest.raises(ValueError):
        primitive.update_states(**impossible_shaped_data)


@pytest.mark.parametrize(
    "nan_data",
    [
        dict(position_1=np.array([10, 10, 10]), radius=np.nan),
        dict(position_1=np.array([np.nan, 10, 10]), radius=10.0),
        dict(position_2=np.array([np.nan, 10, 10]), radius=10.0),
    ],
)
def test_update_states_with_nan_values(nan_data):
    default_data = dict(
        position_1=np.array([0, 0, 0]),
        position_2=np.array([1, 1, 1]),
        radius=1.0,
    )
    primitive = Cylinder.create(default_data)
    with pytest.raises(ValueError) as exc_info:
        primitive.update_states(**nan_data)
    assert "contains NaN" in str(exc_info.value)


def test_cylinder_creator():
    default_data = dict(
        position_1=np.array([0, 0, 0]),
        position_2=np.array([1, 1, 1]),
        radius=1.0,
    )
    primitive = Cylinder.create(default_data)
    old_cylinder = primitive.object
    new_cylinder = primitive._create_cylinder()
    assert new_cylinder is not None
    assert old_cylinder is not new_cylinder
    assert isinstance(new_cylinder, bpy_types.Object)
