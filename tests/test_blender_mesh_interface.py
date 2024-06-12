import bpy
import numpy as np
import pytest

from bsr.geometry import Cylinder, Frustum, Sphere


@pytest.mark.parametrize(
    "primitive",
    [
        Sphere(position=np.array([0, 0, 0]), radius=1.0),
        Cylinder(
            position_1=np.array([0, 0, 0]),
            position_2=np.array([0, 0, 1]),
            radius=1.0,
        ),
        Frustum(
            position_1=np.array([0, 0, 0]),
            position_2=np.array([0, 0, 1]),
            radius_1=1.0,
            radius_2=2.0,
        ),
    ],
)
class TestBlenderMeshInterfaceObjects:
    def test_object_type(self, primitive):
        # TODO : Test .object and return type
        assert isinstance(primitive.object, bpy.types.Object)

    def test_create_method(self, primitive):
        # TODO : Test .create method using .states
        states = primitive.states
        new_object = type(primitive).create(states)
        assert states == new_object.states

    def test_update_states_method(self, primitive):
        # TODO: Test .update_states method and check if the object is updated
        states = primitive.states
        new_object = type(primitive).update_states(
            states
        )  # Use states as pass in?
        assert states != new_object.states


def test_sphere_creator():
    # TODO : Test Sphere._create_sphere method
    new_sphere = Sphere._create_sphere()
    assert new_sphere is not None
    assert isinstance(new_sphere, bpy.types.Object)


# TODO : Add test for Cylinder for full coverage
def test_cylinder_creator():
    new_cylinder = Cylinder._create_cylinder()
    assert new_cylinder is not None
    assert isinstance(new_cylinder, bpy.types.Object)
