from typing import Iterable, Optional

import bpy

from .utilities.singleton import SingletonMeta


class FrameManager(metaclass=SingletonMeta):
    """
    This class provides methods for manipulating the frame of the scene.
    Only one instance exist, which you can access by: bsr.frame_manager.
    """

    def update(self, forward_frame: int = 1) -> None:
        """
        Update the current frame number of the scene.

        Parameters
        ----------
        forward_frame : int, optional
            The number of frames to move forward. The default is 1.
        """
        assert (
            isinstance(forward_frame, int) and forward_frame > 0
        ), "forward_frame must be a positive integer"
        bpy.context.scene.frame_current += forward_frame

    @property
    def frame_current(self) -> int:
        """
        Return the current frame number of the scene.
        """
        return int(bpy.context.scene.frame_current)

    @frame_current.setter
    def frame_current(self, frame: int) -> None:
        """
        Set the current frame number of the scene.

        Parameters
        ----------
        frame : int
            The current frame number of the scene.
        """
        assert (
            isinstance(frame, int) and frame >= 0
        ), "frame must be a positive integer or 0"
        bpy.context.scene.frame_current = frame

    @property
    def frame_start(self) -> int:
        """
        Return the start frame number of the scene.
        """
        return int(bpy.context.scene.frame_start)

    @property
    def frame_end(self) -> int:
        """
        Return the end frame number of the scene.
        """
        return int(bpy.context.scene.frame_end)

    def set_frame_start(self, frame: Optional[int] = None) -> None:
        """
        Set the start frame number of the scene.

        Parameters
        ----------
        frame : int, optional
            The start frame number of the scene. The default is None.
            If None, the current frame number is used.
        """
        if frame is None:
            frame = bpy.context.scene.frame_current
        else:
            assert (
                isinstance(frame, int) and frame >= 0
            ), "frame must be a positive integer or 0"
        bpy.context.scene.frame_start = frame

    def set_frame_end(self, frame: Optional[int] = None) -> None:
        """
        Set the end frame number of the scene.

        Parameters
        ----------
        frame : int, optional
            The end frame number of the scene. The default is None.
            If None, the current frame number is used.
        """
        if frame is None:
            frame = bpy.context.scene.frame_current
        else:
            assert (
                isinstance(frame, int) and frame >= 0
            ), "frame must be a positive integer or 0"
        bpy.context.scene.frame_end = frame

    def set_frame_rate(self, fps: int | float) -> None:
        """
        Set the frame rate of the scene.

        Parameters
        ----------
        fps : float
            The frame rate of the scene. (Frame per second)
        """
        assert isinstance(fps, (int, float)), "fps must be a number"
        assert fps > 0, "fps must be a positive value"
        bpy.context.scene.render.fps = int(fps)
        bpy.context.scene.render.fps_base = int(fps) / fps

    def get_frame_rate(self) -> float:
        """
        Get the frame rate of the scene.

        Returns
        -------
        float
            The frame rate of the scene. (Frame per second)
        """
        fps: float = (
            bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        )
        return fps

    def enumerate(
        self, iterable: Iterable, frame_current: Optional[int] = None
    ):
        """
        Enumerate through the frames of the scene.

        Parameters
        ----------
        iterable : Iterable
            An iterable object to enumerate.
        frame_current : int, optional
            The current frame number of the scene. The default is None.
            If None, the number self.frame_current is used.
        """
        if frame_current is not None:
            self.frame_current = frame_current
        for k, item in enumerate(iterable):
            yield self.frame_current, item
            if k != len(iterable) - 1:
                self.update()  # Update the frame number
            else:
                self.set_frame_end()  # Set the final frame number
