import bpy.types as bpy_types
import numpy as np
import pytest

from bsr.geometry import Sphere


# Sphere-specific tests
@pytest.mark.parametrize(
    "possible_sphere_data",
    [
        dict(position=np.array([10, 10, 10]), radius=10.0),
        dict(radius=10.0),
        dict(position=np.array([10, 10, 10])),
    ],
)
def test_update_states_with_data(possible_sphere_data):
    default_data = dict(position=np.array([0, 0, 0]), radius=1.0)
    primitive = Sphere.create(default_data)
    primitive.update_states(**possible_sphere_data)

    if "position" in possible_sphere_data:
        assert (
            primitive.object.location.x == possible_sphere_data["position"][0]
        )
        assert (
            primitive.object.location.y == possible_sphere_data["position"][1]
        )
        assert (
            primitive.object.location.z == possible_sphere_data["position"][2]
        )
    if "radius" in possible_sphere_data:
        assert primitive.object.scale[0] == possible_sphere_data["radius"]
        assert primitive.object.scale[1] == possible_sphere_data["radius"]
        assert primitive.object.scale[2] == possible_sphere_data["radius"]


@pytest.mark.parametrize(
    "impossible_shaped_data",
    [
        dict(position=np.array([10, 10, 10, 10]), radius=10.0),
        dict(radius=np.array([10, 10, 10])),
        dict(radius=-1),
        dict(radius=0),
        dict(position=np.array([10, 10])),
    ],
)
def test_update_states_with_wrong_shape(impossible_shaped_data):
    default_data = dict(position=np.array([0, 0, 0]), radius=1.0)
    primitive = Sphere.create(default_data)
    with pytest.raises(ValueError):
        primitive.update_states(**impossible_shaped_data)


@pytest.mark.parametrize(
    "nan_data",
    [
        dict(position=np.array([10, 10, 10]), radius=np.nan),
        dict(position=np.array([np.nan, 10, 10]), radius=10.0),
    ],
)
def test_update_states_with_nan_values(nan_data):
    default_data = dict(position=np.array([0, 0, 0]), radius=1.0)
    primitive = Sphere.create(default_data)
    with pytest.raises(ValueError) as exc_info:
        primitive.update_states(**nan_data)
    assert "contains NaN" in str(exc_info.value)


def test_sphere_creator():
    default_data = dict(position=np.array([0, 0, 0]), radius=1.0)
    primitive = Sphere.create(default_data)
    old_sphere = primitive.object
    new_sphere = primitive._create_sphere()
    assert new_sphere is not None
    assert old_sphere is not new_sphere
    assert isinstance(new_sphere, bpy_types.Object)
