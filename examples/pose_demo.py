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


def main(filename: str = "pose_demo"):

    # initial values for frame rate and total time
    frame_rate = 60
    total_time = 5

    # calculates total number of frames in the visualization
    total_frames = frame_rate * total_time

    # clears all mesh objects
    bsr.clear_mesh_objects()

    # initializes pose instance
    pose_object = Pose(
        positions=np.array([1, 0, 0]),
        directors=np.eye(3),
        thickness_ratio=0.1,
    )

    # creates an array of angles from 0 to 360 degrees
    angles = np.linspace(0, 360, total_frames)

    # Set frame start
    bsr.frame_manager.set_frame_start()

    # iterates through each angle
    for angle in angles:

        # defines path of of motion for positions of pose object
        positions = [np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0.0]

        # defines directors of pose object
        d2 = [-np.sin(np.radians(angle)), np.cos(np.radians(angle)), 0.0]
        d3 = [0, 0, 1]
        d1 = np.cross(d2, d3)
        directors = np.column_stack((d1, d2, d3))

        # updates positions and directors of pose object at each keyframe
        pose_object.update_states(
            positions=np.array(positions),
            directors=directors,
        )

        # converts angle to rgb color value at each frame
        color = angle_to_color(angle)

        # updates pose object's colors
        pose_object.update_material(color=color)

        # sets and updates keyframes
        pose_object.set_keyframe(bsr.frame_manager.frame_current)

        if bsr.frame_manager.frame_current == total_frames - 1:
            # Set the final keyframe
            bsr.frame_manager.set_frame_end()
        else:
            # updates frame
            bsr.frame_manager.update()

    # Set the frame rate
    bsr.frame_manager.set_frame_rate(fps=frame_rate)

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
