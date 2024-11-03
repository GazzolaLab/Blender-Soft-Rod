import math

import bpy
import pytest

from bsr.frame import FrameManager


class TestFrameManager:

    def test_frame_manager_singleton(self):
        assert FrameManager() is FrameManager()

    def test_frame_manager_current_frame_setter(self):
        frame_manager = FrameManager()
        frame_manager.frame_current = 10
        assert frame_manager.frame_current == 10

    def test_frame_manager_current_frame_setter_with_wrong_frame(self):
        frame_manager = FrameManager()
        with pytest.raises(AssertionError):
            frame_manager.frame_current = -1

    def test_frame_manager_update(self):
        frame_manager = FrameManager()
        frame_manager.frame_current = 0
        frame_manager.update(10)
        assert frame_manager.frame_current == 10

    def test_frame_manager_update_with_wrong_frame_forward(self):
        frame_manager = FrameManager()
        with pytest.raises(AssertionError):
            frame_manager.update(-1)

    def test_frame_manager_get_set_frame_start(self):
        frame_manager = FrameManager()
        frame_manager.frame_start = 10
        assert bpy.context.scene.frame_start == 10
        assert frame_manager.frame_start == 10

    def test_frame_manager_set_frame_start_with_wrong_frame(self):
        frame_manager = FrameManager()
        frame_start = frame_manager.frame_start
        with pytest.raises(AssertionError):
            frame_manager.frame_start = -1
        assert frame_manager.frame_start == frame_start

    def test_frame_manager_get_set_frame_end(self):
        frame_manager = FrameManager()
        frame_manager.frame_end = 100
        assert bpy.context.scene.frame_end == 100
        assert frame_manager.frame_end == 100

    def test_frame_manager_set_frame_end_with_wrong_frame(self):
        frame_manager = FrameManager()
        frame_end = frame_manager.frame_end
        with pytest.raises(AssertionError):
            frame_manager.frame_end = -1
        assert frame_manager.frame_end == frame_end

    def test_frame_manager_get_set_frame_rate(self):
        frame_manager = FrameManager()
        frame_manager.frame_rate = 30
        assert frame_manager.frame_rate == 30
        frame_manager.frame_rate = 29.97
        assert math.isclose(frame_manager.frame_rate, 29.97, abs_tol=0.001)

    def test_frame_manager_set_frame_rate_with_wrong_frame_rate(self):
        frame_manager = FrameManager()
        frame_rate = frame_manager.frame_rate
        with pytest.raises(AssertionError):
            frame_manager.frame_rate = 0
        assert frame_manager.frame_rate == frame_rate

    def test_frame_manager_enumerate(self):
        frame_manager = FrameManager()
        frame_start = 10
        for frame_current, frame in frame_manager.enumerate(
            range(5), frame_current_init=frame_start
        ):
            assert frame_current == (frame + frame_start)
            assert frame_manager.frame_current == frame_current
        assert frame_manager.frame_end == 14
        assert frame_manager.frame_current == 15
