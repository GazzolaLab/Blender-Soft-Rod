import bpy
import numpy as np
import pytest

from bsr.geometry.primitives.simple import Cylinder, Sphere


class TestBlenderMeshInterfaceObjectsSphere:

    @pytest.fixture(autouse=True)
    def primitive(self):
        Data = dict(position=np.array([0, 0, 0]), radius=1.0)
        return Sphere.create(Data)

    def test_object_type(self, primitive):
        assert isinstance(primitive.object, bpy.types.Object)

    def test_update_states_with_empty_data(self, primitive):
        primitive.update_states()  # Calling empty data should pass
        assert True


class TestBlenderMeshInterfaceObjectsCylinder(
    TestBlenderMeshInterfaceObjectsSphere
):
    @pytest.fixture(autouse=True)
    def primitive(self):
        Data = dict(
            position_1=np.array([0, 0, 0]),
            position_2=np.array([0, 0, 1]),
            radius=1.0,
        )
        return Cylinder.create(Data)


@pytest.mark.parametrize(
    "wrong_key",
    [
        "__wrong_key",
        5,
    ],
)
def test_update_states_warning_message_if_wrong_key_sphere(
    wrong_key,
):
    t = Sphere
    data = {wrong_key: 0, "position": np.array([0, 0, 0]), "radius": 1.0}
    with pytest.warns(UserWarning) as record:
        t.create(data)
    assert (
        f"not used as a part of the state definition"
        in record[0].message.args[0]
    )
    assert str(wrong_key) in record[0].message.args[0]


@pytest.mark.parametrize(
    "wrong_key",
    [
        "__wrong_key",
        5,
    ],
)
def test_update_states_warning_message_if_wrong_key_cylinder(
    wrong_key,
):
    t = Cylinder
    data = {
        wrong_key: 0,
        "position_1": np.array([0, 0, 0]),
        "position_2": np.array([0, 0, 1]),
        "radius": 1.0,
    }
    with pytest.warns(UserWarning) as record:
        t.create(data)
    assert (
        f"not used as a part of the state definition"
        in record[0].message.args[0]
    )
    assert str(wrong_key) in record[0].message.args[0]
