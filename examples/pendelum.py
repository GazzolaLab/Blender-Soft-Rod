import os

import bpy
import numpy as np

import bsr


class PendulumBlender:
    def __init__(self, location, ball_radius):
        self.location = location
        self.ball_radius = ball_radius

    def update(self, position):
        # TODO
        pass


class Pendulum:  # Pendulum simulator
    def __init__(self, length, euler_angles):
        self.length = length
        self.euler_angles = euler_angles
        self.position = self.calculate_position()
        self.velocity = np.array([0.0, 0.0, 0.0])

    def calculate_position(self):
        x = (
            self.length
            * np.cos(self.euler_angles[1])
            * np.cos(self.euler_angles[0])
        )
        y = (
            self.length
            * np.cos(self.euler_angles[1])
            * np.sin(self.euler_angles[0])
        )
        z = self.length * np.sin(self.euler_angles[1])
        return np.array([x, y, z])

    def get_position(self):
        return self.position

    def update(self, dt):
        # Update the velocity
        self.position += self.velocity * dt / 2
        self.velocity += np.array([0, 0, -9.81]) * dt
        self.position += self.velocity * dt / 2


def main():
    bsr.clear_mesh_objects()

    pendulum_length = 0.3
    pendulum_euler_angles = np.array([0.0, 0.0])
    pendulum = Pendulum(
        length=pendulum_length, euler_angles=pendulum_euler_angles
    )
    pendulum_blender = PendulumBlender(
        location=pendulum.get_position(), ball_radius=0.2
    )

    dt = 10 ** (-3)
    framerate = 25
    simulation_ratio = int(1 / framerate / dt)
    time = np.arange(0, 10, dt)

    for k, t in enumerate(time):
        pendulum.update(dt)
        if k % simulation_ratio == 0:
            # Update the location of the pendulum
            pendulum_blender.update(position=pendulum.get_position())

            # # Update the scene
            bpy.context.view_layer.update()

    ### Saving the file ###
    write_filepath = "pendulum.blend"
    bsr.save(filepath=write_filepath)


if __name__ == "__main__":
    main()
