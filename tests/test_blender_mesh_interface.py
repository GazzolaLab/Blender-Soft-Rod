import bpy
import numpy as np
import pytest

from bsr.geometry import Cylinder, Frustum, Sphere


@pytest.mark.parametrize(
    "primitive_type_and_data",
    [
        (Sphere, dict(position=np.array([0, 0, 0]), radius=1.0)),
        (
            Cylinder,
            dict(
                position_1=np.array([0, 0, 0]),
                position_2=np.array([0, 0, 1]),
                radius=1.0,
            ),
        ),
    ],
)
class TestBlenderMeshInterfaceObjects:
    @pytest.fixture(scope="function")
    def primitive(self, primitive_type_and_data):
        primitive_type, data = primitive_type_and_data
        return primitive_type(**data)

    def test_object_type(self, primitive):
        assert isinstance(primitive.object, bpy.types.Object)

    def test_create_method(self, primitive_type_and_data):
        primitive_type, data = primitive_type_and_data
        new_object = primitive_type.create(data)

        assert new_object is not None

    def test_update_states_with_empty_data(self, primitive):
        primitive.update_states()  # Calling empty data should pass

    @pytest.mark.parametrize(
        "wrong_key",
        [
            "__wrong_key",
            5,
        ],
    )
    def test_update_states_warning_message_if_wrong_key(
        self, primitive_type_and_data, wrong_key
    ):
        t, data = primitive_type_and_data
        data.update({wrong_key: 0})
        with pytest.warns(UserWarning) as record:
            t.create(data)
        assert (
            f"not used as a part of the state definition"
            in record[0].message.args[0]
        )
        assert str(wrong_key) in record[0].message.args[0]
