import numpy as np

import bsr
from bsr.geometry.composite.pose import Pose


def main(
    filename: str = "camera_movement",
    frame_rate: int = 60,
    total_time: float = 5.0,
    camera_location_height: float = 1.0,
    camera_orbiting_radius: float = 1.0,
):

    # Create a new scene
    bsr.clear_mesh_objects()

    # Set the camera film background to transparent
    bsr.camera.set_film_transparent()

    # Set the camera look at location
    bsr.camera.look_at = np.array([0.0, 0.0, 0.0])

    # Set a frame at the origin
    _ = Pose(
        position=np.zeros(3),
        directors=np.identity(3),
        unit_length=0.25,
    )

    # Set the current frame number
    bsr.frame_manager.frame_current = 0

    # Set the initial keyframe number
    bsr.frame_manager.set_frame_start()

    # Set the camera orbiting keyframes
    angles = np.linspace(0.0, 360.0, frame_rate * total_time, endpoint=False)
    for k, angle in enumerate(angles):

        # Set the camera location
        bsr.camera.location = np.array(
            [
                camera_orbiting_radius * np.cos(np.radians(angle)),
                camera_orbiting_radius * np.sin(np.radians(angle)),
                camera_location_height,
            ]
        )

        # Update the keyframe
        bsr.camera.set_keyframe(bsr.frame_manager.frame_current)

        if k != len(angles) - 1:
            # Update the keyframe number
            bsr.frame_manager.update()
        else:
            # Set the final keyframe number
            bsr.frame_manager.set_frame_end(bsr.frame_manager.frame_current)

    # Set the frame rate
    bsr.frame_manager.set_frame_rate(fps=frame_rate)

    # Set the view distance
    bsr.set_view_distance(distance=5)

    # Deslect all objects
    bsr.deselect_all()

    # Select the camera object
    bsr.camera.select()

    # Set the render file path
    bsr.camera.set_file_path("render/" + filename)

    # set resolution
    bsr.camera.set_resolution(1920, 1080)

    # render the scene
    bsr.camera.render(
        frames=np.arange(
            bsr.frame_manager.frame_start, bsr.frame_manager.frame_end + 1
        )
    )

    # Save as .blend file
    bsr.save(filename + ".blend")


if __name__ == "__main__":
    main()
