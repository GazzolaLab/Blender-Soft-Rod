import bpy
import numpy as np

from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class Light(KeyFrameControlMixin):
    """
    This class provides methods for manipulating the light of the scene.
    """

    def __init__(self, name: str = "Light") -> None:
        """
        Constructor for light.
        """
        self.name = name

    @property
    def _light(self) -> bpy.types.Object:
        """
        Return the light object.
        """
        return bpy.data.objects[self.name]

    def select(self) -> None:
        """
        Select the light object.
        """
        bpy.context.view_layer.objects.active = self._light

    @property
    def location(self) -> np.ndarray:
        """
        Return the current location of the light.
        """
        return np.array(self._light.location)

    @location.setter
    def location(self, location: np.ndarray) -> None:
        """
        Set the location of the light.

        Parameters
        ----------
        location : np.ndarray
        """
        self._light.location = location

    def update_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self._light.keyframe_insert(data_path="location", frame=keyframe)
