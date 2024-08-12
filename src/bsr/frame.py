from typing import Optional

import bpy

from .utilities.singleton import Singleton

class FrameManager(Singleton):
    """
    This class provides methods for manipulating the frame of the scene.
    Only one instance exist, which you can access by: bsr.frame.
    """

    def __init__(self, frame: int = 0):
        """
        Constructor for frame manager.
            
        Parameters
        ----------
        frame : int, optional
            The initial frame of the scene. The default is 0.
        """
        self.__frame = frame

    def update(self, forwardframe: int = 1) -> None:
        assert (
            isinstance(forwardframe, int) and forwardframe > 0
        ), "forwardframe must be a positive integer"
        self.__frame += forwardframe

    @property
    def current_frame(self) -> int:
        """
        Return the current frame number of the scene.
        """
        return self.__frame

    @current_frame.setter
    def current_frame(self, frame: int) -> None:
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
        self.__frame = frame

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
            frame = self.__frame
        else:
            assert (
                isinstance(frame, int) and frame >= 0
            ), "frame must be a positive integer or 0"
        bpy.context.scene.frame_end = frame
