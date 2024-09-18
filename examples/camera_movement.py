import numpy as np

import bsr
from bsr.geometry.composite.pose import Pose


def main(filename: str = "camera_movement"):

    frame_rate = 60
    total_time = 5

    camera_heigh = 1.0
    camera_radius = 1.0

    bsr.clear_mesh_objects()

    bsr.camera.look_at = np.array([0.0, 0.0, 0.0])
    bsr.camera.set_film_transparent()

    pose = Pose(
        position=np.zeros(3),
        directors=np.identity(3),
        unit_length=0.25,
    )
    angles = np.linspace(0.0, 360.0, frame_rate * total_time, endpoint=False)

    # Set the initial keyframe number
    bsr.frame_manager.set_frame_start(0)

    # Set the current frame number
    bsr.frame_manager.frame_current = 0

    for k, angle in enumerate(angles):
        bsr.camera.location = np.array(
            [
                camera_radius * np.cos(np.radians(angle)),
                camera_radius * np.sin(np.radians(angle)),
                camera_heigh,
            ]
        )
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
