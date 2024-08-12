from abc import ABC, abstractmethod

import bpy

from bsr.geometry.protocol import BlenderMeshInterfaceProtocol


# Base class
class KeyFrameControlMixin(ABC):
    """
    This mixin class provides methods for manipulating keyframes.
    By adding this mixin, the class will conform to the BlenderKeyframeManipulateProtocol.
    Otherwise, each methods must be implemented in the class.
    """

    def clear_animation(self: BlenderMeshInterfaceProtocol) -> None:
        """
        Clear all keyframes of the object.
        """
        self.object.animation_data_clear()
        self.object.animation_data_create()

    @abstractmethod
    def set_keyframe(self: BlenderMeshInterfaceProtocol, keyframe: int) -> None:
        raise NotImplementedError
