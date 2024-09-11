import numpy as np

import bsr


def main(filename: str = "camera_movement"):

    frame_rate = 60
    total_time = 5

    camera_heigh = 1.0
    camera_radius = 1.0

    bsr.clear_mesh_objects()

    bsr.camera.look_at = np.array([0.0, 0.0, 0.0])

    for angle in np.linspace(0.0, 360.0, frame_rate * total_time + 1):
        bsr.camera.location = np.array(
            [
                camera_radius * np.cos(np.radians(angle)),
                camera_radius * np.sin(np.radians(angle)),
                camera_heigh,
            ]
        )
        bsr.camera.set_keyframe(bsr.frame_manager.current_frame)
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
