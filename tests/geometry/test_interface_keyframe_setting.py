import math

import bpy
import numpy as np
import pytest

from bsr.geometry import Cylinder, Sphere


def get_keyframes(obj_list):
    keyframes = []
    for obj in obj_list:
        animation_data = obj.animation_data
        if animation_data is not None and animation_data.action is not None:
            for fcurve in animation_data.action.fcurves:
                for keyframe in fcurve.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append(math.ceil(x))
    return keyframes


def count_number_of_keyframes_action(obj):
    action = obj.animation_data.action
    if action is None:
        return 0
    else:
        return len(action.fcurves[0].keyframe_points)


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
def test_set_keyframe_count_for_primitive(primitive):
    primitive.set_keyframe(1)
    assert count_number_of_keyframes_action(primitive.object) == 1

    primitive.set_keyframe(2)
    assert count_number_of_keyframes_action(primitive.object) == 2

    # Setting keyfrome at the same frame should not increase the number of keyframes:
    primitive.set_keyframe(2)
    assert count_number_of_keyframes_action(primitive.object) == 2

    primitive.clear_animation()
    assert count_number_of_keyframes_action(primitive.object) == 0

    primitive.set_keyframe(1)
    assert count_number_of_keyframes_action(primitive.object) == 1

    # Clear the test
    primitive.clear_animation()
