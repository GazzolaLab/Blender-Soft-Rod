from typing import Optional

import bpy


class Frame:
    def __init__(self, frame: int = 0):
        self.__frame = frame

    def update(self, forwardframe: int = 1) -> None:
        assert (
            isinstance(forwardframe, int) and forwardframe > 0
        ), "forwardframe must be a positive integer"
        self.__frame += forwardframe

    @property
    def current_frame(self) -> int:
        return self.__frame

    @current_frame.setter
    def current_frame(self, frame: int) -> None:
        assert (
            isinstance(frame, int) and frame >= 0
        ), "frame must be a positive integer or 0"
        self.__frame = frame

    def set_frame_end(self, frame: Optional[int] = None) -> None:
        if frame is None:
            frame = self.__frame
        else:
            assert (
                isinstance(frame, int) and frame >= 0
            ), "frame must be a positive integer or 0"
        bpy.context.scene.frame_end = frame
