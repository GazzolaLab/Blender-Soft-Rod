import numpy as np

import bsr
from bsr import Pose


def main(
    filename: str = "camera_movement",
    frame_rate: int = 60,
    total_time: float = 5.0,
    camera_location_height: float = 1.0,
    camera_orbiting_radius: float = 1.0,
):

    # Clear all mesh objects in the new scene
    bsr.clear_mesh_objects()

    # Set the camera film background to transparent
    bsr.camera.set_film_transparent()

    # Set the render file path
    bsr.camera.set_file_path(filename + "/frame")

    # Set resolution
    bsr.camera.set_resolution(1920, 1080)

    # Set the camera look at location
    bsr.camera.look_at = np.array([0.0, 0.0, 0.0])

    # Set a pose at the origin
    _ = Pose(
        positions=np.zeros(3),
        directors=np.identity(3),
        unit_length=0.25,
    )

    # Set the camera orbiting angles
    angles = np.linspace(
        0.0, 360.0, int(frame_rate * total_time), endpoint=False
    )

    # Set the initial frame
    frame_start = 0
    bsr.frame_manager.frame_start = frame_start

    for frame_current, angle in bsr.frame_manager.enumerate(
        angles, frame_current=frame_start
    ):

        # Set the camera location
        bsr.camera.location = np.array(
            [
                camera_orbiting_radius * np.cos(np.radians(angle)),
                camera_orbiting_radius * np.sin(np.radians(angle)),
                camera_location_height,
            ]
        )

        # Set and update the camera in current frame
        bsr.camera.set_keyframe(frame_current)

    # Set the frame rate
    bsr.frame_manager.frame_rate = frame_rate

    # Set the view distance
    bsr.set_view_distance(distance=5)

    # Deselect all objects
    bsr.deselect_all()

    # Select the camera object
    bsr.camera.select()

    # Render the scene
    bsr.camera.render(
        frames=np.arange(
            bsr.frame_manager.frame_start, bsr.frame_manager.frame_end + 1
        )
    )

    # Save as .blend file
    bsr.save(filename + ".blend")


if __name__ == "__main__":
    main()
    print("\n\nTo convert the frames into a video, run the following command:")
    print(
        r"ffmpeg -threads 8 -r 60 -i camera_movement/frame_%03d.png -b:v 90M -c:v prores -pix_fmt yuva444p10le camera_movement.mov"
    )
