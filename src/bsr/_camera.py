from typing import Optional, Union

from pathlib import Path

import bpy
import numpy as np
from numpy.typing import NDArray

from bsr.frame import FrameManager
from bsr.tools.keyframe_mixin import KeyFrameControlMixin


class Camera(KeyFrameControlMixin):
    """
    This class provides methods for manipulating the camera of the scene.
    """

    def __init__(self, name: str = "Camera") -> None:
        """
        Constructor for camera.
        """
        self._name = name
        self._sky = np.array([0.0, 0.0, 1.0])
        self.__look_at_location: Optional[np.ndarray] = None
        self.__render_folder_path: Optional[Path] = None
        self.__render_file_name: Optional[str] = None
        self.__render_file_type: Optional[str] = None

    @property
    def _camera(self) -> bpy.types.Object:
        """
        Return the camera object.
        """
        return bpy.data.objects[self._name]

    def select(self) -> None:
        """
        Select the camera object.
        """
        bpy.context.view_layer.objects.active = self._camera

    def set_keyframe(self, keyframe: int) -> None:
        """
        Sets a keyframe at the given frame.

        Parameters
        ----------
        keyframe : int
        """
        self._camera.keyframe_insert(data_path="location", frame=keyframe)
        self._camera.keyframe_insert(data_path="rotation_euler", frame=keyframe)

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
        return np.array(self._camera.location)

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
        self._camera.location = location
        if self.__look_at_location is not None:
            self._camera.matrix_world = self.compute_matrix_world(
                location=self.location,
                direction=self.__look_at_location - self.location,
                sky=self._sky,
            )

    @property
    def orientation(self) -> np.ndarray:
        """
        Return the current orientation of the camera.
        """
        return np.array(self._camera.matrix_world)[:3, :3]

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
        self._camera.matrix_world = self.compute_matrix_world(
            location=self.location,
            direction=self.__look_at_location - self.location,
            sky=self._sky,
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
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)
        return np.array(
            [[*right, 0.0], [*up, 0.0], [*(-direction), 0.0], [*location, 1.0]]
        )

    def rotate(self, angle: float) -> None:
        """
        Rotate the camera around the look-at-axis.

        Parameters
        ----------
        angle : float
            The angle (degree) to rotate the camera.
        """
        assert isinstance(angle, (int, float)), "angle must be a number"
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array(
            [
                [np.cos(angle_rad), -np.sin(angle_rad), 0],
                [np.sin(angle_rad), np.cos(angle_rad), 0],
                [0, 0, 1],
            ]
        )
        orientation = self.orientation @ rotation_matrix
        self._camera.matrix_world = np.array(
            [
                [*orientation[:, 0], 0.0],
                [*orientation[:, 1], 0.0],
                [*orientation[:, 2], 0.0],
                [*self.location, 1.0],
            ]
        )

    def set_resolution(self, width: int, height: int) -> None:
        """
        Set the resolution of the camera.

        Parameters
        ----------
        width : int
            The width of the camera.
        height : int
            The height of the camera.
        """
        assert isinstance(width, int), "width must be an integer"
        assert isinstance(height, int), "height must be an integer"
        self.select()
        bpy.context.scene.render.resolution_x = width
        bpy.context.scene.render.resolution_y = height

    def set_file_path(
        self,
        file_name: Union[str, Path],
        folder_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Set the file path for rendering.

        Parameters
        ----------
        file_name : str or Path
            The name of the file.
        folder_path : str or Path, optional
            The folder path. Default is None.
        """
        # check the type of arguments
        assert isinstance(
            file_name, (str, Path)
        ), f"file_name {file_name} must be a string or Path."
        file_name = Path(file_name)
        if folder_path is not None:
            assert isinstance(
                folder_path, (str, Path)
            ), f"folder_path {folder_path} must be a string or Path."
            folder_path = Path(folder_path)
        else:
            folder_path = Path()

        # check the file name extension
        if file_name.suffix != "":
            assert file_name.suffix in [
                ".png"
            ], f"file_name {file_name} must have a .png extension."

        # check if the file_name has a parent folder
        if len(file_name.parts) > 1:
            assert (
                file_name.parent != folder_path
            ), "parent folder of file_name conflicts with folder_path. Please only provide file_name."
            folder_path = Path(file_name.parent)
            file_name = Path(file_name.name)

        # check if the folder path exists
        if not folder_path.exists():
            print(f"folder_path {folder_path} does not exist")
            print(f"Creating folder_path {folder_path}")
            folder_path.mkdir(parents=True)

        self.__render_file_type = file_name.suffix
        self.__render_file_name = file_name.stem
        self.__render_folder_path = folder_path

    def get_file_path(
        self,
        frame: Optional[int] = None,
        number_of_digits: int = 4,
    ) -> str:
        """
        Get the file path for rendering.

        Parameters
        ----------
        frame : int, optional
            The frame number. Default is None.
        number_of_digits : int, optional
            The number of digits for the frame number. Default is 4.

        Returns
        -------
        str
            The file path for rendering.
        """
        # check if the render file path is set
        assert (
            self.__render_folder_path is not None
        ), "render file path is not set"
        assert (
            self.__render_file_name is not None
        ), "render file name is not set"
        assert (
            self.__render_file_type is not None
        ), "render file type is not set"

        if frame is None:
            file_path = self.__render_folder_path / Path(
                self.__render_file_name + self.__render_file_type
            )
        else:
            assert isinstance(frame, int), f"frame {frame} must be an integer"
            file_path = self.__render_folder_path / Path(
                self.__render_file_name
                + f"_{frame:0{number_of_digits}d}"
                + self.__render_file_type
            )
        return str(file_path)

    def render(
        self, frames: Optional[Union[int, list, tuple, NDArray]] = None
    ) -> None:
        """
        Render the scene.

        Parameters
        ----------
        frames : int, list, tuple or np.array, optional
            The frames to render. Default is None, which means rendering the current frame only.
        """
        frame_manager = FrameManager()

        # check the type of the argument: frames
        if frames is None:
            bpy.context.scene.render.filepath = self.get_file_path()
            bpy.ops.render.render(write_still=True)
            return

        if isinstance(frames, int):
            frames = (frames,)
        elif isinstance(frames, (list, tuple)):
            assert all(
                isinstance(frame, int) for frame in frames
            ), "frames must be a list or tuple of integers"
            frames = tuple(frames)
        elif isinstance(frames, np.ndarray):
            assert frames.ndim == 1, "frames must be a 1D array"
            assert np.issubdtype(
                frames.dtype, np.integer
            ), "frames must be an integer array"
            frames = tuple(int(frame) for frame in frames)
        else:
            raise ValueError(
                "frames must be an integer, list, tuple or 1D numpy array"
            )

        frame_current = frame_manager.frame_current

        max_frame_digits = len(str(np.max(frames)))
        for frame in frames:
            frame_manager.frame_current = frame
            bpy.context.scene.render.filepath = self.get_file_path(
                frame, max_frame_digits
            )
            bpy.ops.render.render(write_still=True)

        frame_manager.frame_current = frame_current
