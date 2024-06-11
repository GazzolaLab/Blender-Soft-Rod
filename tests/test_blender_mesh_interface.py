import bpy
import numpy as np
import pytest

from bsr.geometry import Cylinder, Sphere


@pytest.mark.parametrize(
    "primitive",
    [
        Sphere(position=np.array([0, 0, 0]), radius=1.0),
        Cylinder(
            position_1=np.array([0, 0, 0]),
            position_2=np.array([0, 0, 1]),
            radius=1.0,
        ),
    ],
)
class TestBlenderMeshInterfaceObjects:
    def test_object_type(self, primitive):
        # TODO : Test .object and return type
        assert isinstance(primitive.object, bpy.types.Object)

    def test_create_method(self, primitive):
        # TODO : Test .create method using .states
        if isinstance(primitive, Sphere):
            states = {"position": np.array([0, 0, 0]), "radius": 1.0}
            created = Sphere.create(states)
            assert isinstance(created, Sphere)
        elif isinstance(primitive, Cylinder):
            states = {
                "position_1": np.array([0, 0, 0]),
                "position_2": np.array([0, 0, 1]),
                "radius": 1.0,
            }
            created = Cylinder.create(states)
            assert isinstance(created, Cylinder)
        else:
            raise TypeError(f"Unsupported type: {type(primitive)}")

    def test_update_states_method(self, primitive):
        # TODO: Test .update_states method and check if the object is updated
        assert False


def test_sphere_creator():
    # TODO : Test Sphere._create_sphere method
    assert False


# TODO : Add test for Cylinder for full coverage
