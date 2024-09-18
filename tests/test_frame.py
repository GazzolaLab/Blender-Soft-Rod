import math

import bpy
import pytest

from bsr.frame import FrameManager


class TestFrameManager:

    def test_frame_manager_singleton(self):
        assert FrameManager() is FrameManager()

    def test_frame_manager_current_frame(self):
        frame_manager = FrameManager()
        assert frame_manager.frame_current == 0

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

    def test_frame_manager_update_with_wrong_forwardframe(self):
        frame_manager = FrameManager()
        with pytest.raises(AssertionError):
            frame_manager.update(-1)

    def test_frame_manager_set_frame_end(self):
        frame_manager = FrameManager()
        frame_manager.set_frame_end(100)
        assert bpy.context.scene.frame_end == 100

    def test_frame_manager_set_frame_end_with_none(self):
        frame_manager = FrameManager()
        frame_manager.frame_current = 0
        frame_manager.update(250)
        frame_manager.set_frame_end()
        assert bpy.context.scene.frame_end == 250

    def test_frame_manager_set_frame_end_with_wrong_frame(self):
        frame_manager = FrameManager()
        with pytest.raises(AssertionError):
            frame_manager.set_frame_end(-1)

    def test_frame_manager_get_set_frame_rate(self):
        frame_manager = FrameManager()
        frame_manager.set_frame_rate(30)
        assert frame_manager.get_frame_rate() == 30
        frame_manager.set_frame_rate(29.97)
        assert math.isclose(
            frame_manager.get_frame_rate(), 29.97, abs_tol=0.001
        )
