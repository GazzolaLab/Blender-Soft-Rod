import numpy as np

import bsr
from bsr import Pose


def angle_to_color(angle: float) -> np.ndarray:
    """
    Convert angle to RGB color value

    Parameters
    ----------
    angle : float
        The angle in degrees

    Returns
    -------
    np.ndarray
        The RGBA color value
    """

    # Reset angle range to 0-360 degrees
    angle = angle % 360

    # Convert angle to radians
    rad = np.radians(angle)

    # Calculate RGB values
    if angle < 120:
        rad = 3 * rad / 2
        r = 0.5 + np.sin(rad) / 2
        g = 0.5 - np.sin(rad) / 2
        b = 0.5 - np.sin(rad) / 2
    elif angle < 240:
        rad = rad - 2 * np.pi / 3
        rad = 3 * rad / 2
        r = 0.5 - np.sin(rad) / 2
        g = 0.5 + np.sin(rad) / 2
        b = 0.5 - np.sin(rad) / 2
    else:
        rad = rad - 2 * np.pi / 3 * 2
        rad = 3 * rad / 2
        r = 0.5 - np.sin(rad) / 2
        g = 0.5 - np.sin(rad) / 2
        b = 0.5 + np.sin(rad) / 2

    # Return RGBA numpy array
    return np.array([r, g, b, 1.0])


def main(
    filename: str = "pose_demo",
    frame_rate: int = 60,
    total_time: float = 5.0,
):

    # Clear all mesh objects in the new scene
    bsr.clear_mesh_objects()

    # Initialize pose instance
    pose = Pose(
        positions=np.array([1, 0, 0]),
        directors=np.eye(3),
        thickness_ratio=0.1,
    )

    # Create an array of angles from 0 to 360 degrees
    angles = np.linspace(
        0.0, 360.0, int(frame_rate * total_time), endpoint=False
    )

    # Set the initial frame
    frame_start = 0
    bsr.frame_manager.frame_start = frame_start

    for frame_current, angle in bsr.frame_manager.enumerate(
        angles, frame_current_init=frame_start
    ):

        # Define path of of motion for positions of pose object
        positions = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.0]

        # Define directors of pose object
        d2 = [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0.0]
        d3 = [0, 0, 1]
        d1 = np.cross(d2, d3)
        directors = np.column_stack((d1, d2, d3))

        # Update positions and directors of pose object at each keyframe
        pose.update_states(
            positions=np.array(positions),
            directors=directors,
        )

        # Convert angle to rgb color value at each frame
        color = angle_to_color(angle)

        # Update pose object's colors
        pose.update_material(color=color)

        # Set and update the pose in current frame
        pose.set_keyframe(frame_current)

    # Set the frame rate
    bsr.frame_manager.frame_rate = frame_rate

    # Set the view distance
    bsr.set_view_distance(distance=5)

    # Deselect all objects
    bsr.deselect_all()

    # Select the camera object
    bsr.camera.select()

    # Save as .blend file
    bsr.save(filename + ".blend")


if __name__ == "__main__":
    main()
