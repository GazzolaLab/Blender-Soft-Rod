import bpy
import numpy as np

import bsr
from bsr import Pose


def angle_to_color(angle: float) -> np.ndarray:
    # Normalize angle to 0-360 range
    angle = angle % 360

    # Convert angle to radians
    rad = np.radians(angle)

    # Calculate RGB values
    if angle < 120:
        r = np.cos(rad) / 2 + 0.5
        g = np.sin(rad) / 2 + 0.5
        b = 0
    elif angle < 240:
        r = 0
        g = np.cos(rad - 2 * np.pi / 3) / 2 + 0.5
        b = np.sin(rad - 2 * np.pi / 3) / 2 + 0.5
    else:
        r = np.sin(rad - 4 * np.pi / 3) / 2 + 0.5
        g = 0
        b = np.cos(rad - 4 * np.pi / 3) / 2 + 0.5

    # Return RGBA numpy array
    return np.array([r, g, b, 1.0])


def main(filename: str = "pose_demo5"):

    # initial values for frame rate and period
    frame_rate = 60
    total_time = 5

    # calculates total number of frames in the visualization
    total_frames = frame_rate * total_time

    # clears all mesh objects
    bsr.clear_mesh_objects()
    bsr.frame_manager.set_frame_start()

    # intializes pose instance and angle
    pose_object = Pose(
        positions=np.array([1, 0, 0]), directors=np.eye(3), thickness_ratio=0.1
    )
    theta = 0

    # iterates through each frame in total time duration
    for frame in range(total_frames):
        theta = 2 * np.pi * frame / total_frames

        # defines path of of motion for positions of pose object
        new_positions = np.array([np.cos(theta), np.sin(theta), 0])
        pose_object.positions = new_positions

        # defines directors of pose object
        d2 = np.array([-np.sin(theta), np.cos(theta), 0])
        d3 = np.array([0, 0, 1])
        d1 = np.cross(d2, d3)
        new_directors = np.column_stack((d1, d2, d3))
        pose_object.directors = new_directors

        # updates positions and directors of pose object at each keyframe
        pose_object.update_states(new_positions, new_directors)

        # converts angle to rgb color value at each frame
        color = angle_to_color(np.degrees(theta))

        # updates pose object's colors
        pose_object.update_material(color=color)

        # sets and updates keyframes
        pose_object.set_keyframe(frame)
        bsr.frame_manager.update()

    # Set the final keyframe number
    bsr.frame_manager.set_frame_end()

    # Set the frame rate
    bsr.frame_manager.set_frame_rate(fps=frame_rate)

    # Set the view distance
    bsr.set_view_distance(distance=5)

    # Deslect all objects
    bsr.deselect_all()

    # Select the camera object
    bsr.camera.select()

    # Save as .blend file
    bsr.save(filename + ".blend")


if __name__ == "__main__":
    main()
