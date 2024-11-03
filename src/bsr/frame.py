from typing import Iterable, Optional

import bpy

from .utilities.singleton import SingletonMeta


class FrameManager(metaclass=SingletonMeta):
    """
    This class provides methods for manipulating the frame of the scene.
    Only one instance exist, which you can access by: bsr.frame_manager.
    """

    def update(self, frame_forward: int = 1) -> None:
        """
        Update the current frame number of the scene.

        Parameters
        ----------
        frame_forward : int, optional
            The number of frames to move forward. The default is 1.
        """
        assert (
            isinstance(frame_forward, int) and frame_forward > 0
        ), "frame_forward must be a positive integer"
        bpy.context.scene.frame_current += frame_forward

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

        Returns
        -------
        int
            The start frame number of the scene.
        """
        frame_start = int(bpy.context.scene.frame_start)
        return frame_start

    @frame_start.setter
    def frame_start(self, frame: int) -> None:
        """
        Set the start frame number of the scene.

        Parameters
        ----------
        frame : int
            The start frame number of the scene.
        """
        assert (
            isinstance(frame, int) and frame >= 0
        ), "frame must be a nonnegative integer"
        bpy.context.scene.frame_start = frame

    @property
    def frame_end(self) -> int:
        """
        Return the end frame number of the scene.

        Returns
        -------
        int
            The end frame number of the scene.
        """
        frame_end = int(bpy.context.scene.frame_end)
        return frame_end

    @frame_end.setter
    def frame_end(self, frame: int) -> None:
        """
        Set the end frame number of the scene.

        Parameters
        ----------
        frame : int
            The end frame number of the scene.
        """
        assert (
            isinstance(frame, int) and frame >= 0
        ), "frame must be a nonnegative integer"
        bpy.context.scene.frame_end = frame

    @property
    def frame_rate(self) -> float:
        """
        Return the frame rate of the scene.

        Returns
        -------
        float
            The frame rate of the scene. (Frame per second)
        """
        fps: float = (
            bpy.context.scene.render.fps / bpy.context.scene.render.fps_base
        )
        return fps

    @frame_rate.setter
    def frame_rate(self, fps: int | float) -> None:
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

    def enumerate(
        self, iterable: Iterable, frame_current_init: Optional[int] = None
    ) -> Iterable:
        """
        Enumerate through the frames of the scene.

        Parameters
        ----------
        iterable : Iterable
            An iterable object to enumerate.
        frame_current_init : int, optional
            The initial current frame number of the scene. The default is None.
            If None, the number self.frame_current is used.
        """
        if frame_current_init is not None:
            assert (
                isinstance(frame_current_init, int) and frame_current_init >= 0
            ), "frame_current_init must be a nonnegative integer"
            self.frame_current = frame_current_init
        for item in iterable:
            yield self.frame_current, item
            self.update()  # Update the frame number
        self.frame_end = self.frame_current - 1  # Set the final frame number
