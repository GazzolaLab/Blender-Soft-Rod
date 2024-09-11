from typing import Optional

import bpy
import numpy as np

from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class CameraManager(KeyFrameControlMixin):
    """
    This class provides methods for manipulating the camera of the scene.
    """

    def __init__(self, name: str = "Camera") -> None:
        """
        Constructor for camera manager.
        """
        self.name = name
        self.__look_at_location: Optional[np.ndarray] = None
        self.__sky = np.array([0.0, 0.0, 1.0])

    @property
    def camera(self) -> bpy.types.Object:
        """
        Return the camera object.
        """
        return bpy.data.objects[self.name]

    def select(self) -> None:
        """
        Select the camera object.
        """
        bpy.context.view_layer.objects.active = self.camera

    def set_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self.camera.keyframe_insert(data_path="location", frame=keyframe)
        self.camera.keyframe_insert(data_path="rotation_euler", frame=keyframe)

    def set_film_transparent(self, transparent: bool = True) -> None:
        """
        Set the film transparency for rendering.

        Parameters
        ----------
        transparent : bool, optional
            Whether the film should be transparent. Default is True.
        """
        bpy.context.scene.render.film_transparent = transparent

    @property
    def is_film_transparent(self) -> bool:
        """
        Check if the film is set to transparent.

        Returns
        -------
        bool
            True if the film is transparent, False otherwise.
        """
        film_transparent: bool = bpy.context.scene.render.film_transparent
        return film_transparent

    @property
    def location(self) -> np.ndarray:
        """
        Return the current location of the camera.
        """
        return np.array(self.camera.location)

    @location.setter
    def location(self, location: np.ndarray) -> None:
        """
        Set the location of the camera. If the look at location is set, the camera will be rotated to look at that location.

        Parameters
        ----------
        location : np.array
            The location of the camera.
        """
        assert isinstance(
            location, np.ndarray
        ), "location must be a numpy array"
        assert len(location) == 3, "location must have 3 elements"
        self.camera.location = location
        if self.__look_at_location is not None:
            self.camera.matrix_world = self.compute_matrix_world(
                location=self.location,
                direction=self.__look_at_location - self.location,
                sky=self.__sky,
            )

    @property
    def look_at(self) -> Optional[np.ndarray]:
        """
        Return the location the camera is looking at.
        """
        return self.__look_at_location

    @look_at.setter
    def look_at(self, location: np.ndarray) -> None:
        """
        Set the direction the camera is looking at.

        Parameters
        ----------
        location : np.array
            The direction the camera is looking at.
        """
        assert isinstance(
            location, np.ndarray
        ), "location must be a numpy array"
        assert len(location) == 3, "location must have 3 elements"
        assert (
            np.allclose(self.location, location) == False
        ), "camera and look at location must be different"

        self.__look_at_location = location
        self.camera.matrix_world = self.compute_matrix_world(
            location=self.location,
            direction=self.__look_at_location - self.location,
            sky=self.__sky,
        )

    @staticmethod
    def compute_matrix_world(
        location: np.ndarray, direction: np.ndarray, sky: np.ndarray
    ) -> np.ndarray:
        """
        Compute the world matrix of the camera.

        Parameters
        ----------
        location : np.array
            The location of the camera.
        direction : np.array
            The direction the camera is looking at.
        sky : np.array
            The sky direction of the camera. (unit vector)

        Returns
        -------
        np.array
            The world matrix of the camera.
        """
        assert isinstance(
            location, np.ndarray
        ), "location must be a numpy array"
        assert len(location) == 3, "location must have 3 elements"
        assert isinstance(
            direction, np.ndarray
        ), "direction must be a numpy array"
        assert len(direction) == 3, "direction must have 3 elements"
        assert isinstance(sky, np.ndarray), "sky must be a numpy array"
        assert len(sky) == 3, "sky must have 3 elements"
        assert np.linalg.norm(sky) == 1, "sky must be a unit vector"

        direction = direction / np.linalg.norm(direction)
        right = np.cross(direction, sky)
        up = np.cross(right, direction)

        return np.array(
            [[*right, 0.0], [*up, 0.0], [*(-direction), 0.0], [*location, 1.0]]
        )
