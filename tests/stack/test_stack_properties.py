import pytest

from bsr.stack import BaseStack


def test_object_property():
    stack = BaseStack()
    stack._objs = [1, 2, 3]
    assert stack.object == [1, 2, 3]


def test_set_keyframe(mocker):
    stack = BaseStack()
    mock_rod = mocker.Mock()
    n_repeat = 3
    val = 5
    stack._objs = [mock_rod] * n_repeat
    stack.set_keyframe(val)

    mock_rod.set_keyframe.assert_called()
    assert mock_rod.set_keyframe.call_count == n_repeat
    mock_rod.set_keyframe.assert_called_with(val)
