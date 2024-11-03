import os

import bpy
import numpy as np

import bsr
from bsr.geometry import Cylinder, Sphere

g = 9.81  # m/s^2 acceleration due to gravity


class PendulumBlender:
    def __init__(self, location, ball_radius, cylinder_radius=0.1):
        self.sphere = Sphere(position=location, radius=ball_radius)
        self.cylinder = Cylinder(
            position_1=np.array([0, 0, 0]),
            position_2=location,
            radius=cylinder_radius,
        )

    def update(self, position):
        self.sphere.update_states(position=position)
        self.cylinder.update_states(
            position_1=np.array([0, 0, 0]), position_2=position
        )

    def set_keyframe(self, keyframe):
        self.sphere.set_keyframe(keyframe)
        self.cylinder.set_keyframe(keyframe)


class Pendulum:  # Pendulum simulator
    def __init__(self, length):
        self.length = length
        self.euler_angles = np.array([0.3, 0.2])
        self.angular_velocity = np.array([0.01, 0.00])

    @property
    def position(self):
        return self.calculate_position()

    def calculate_position(self):
        phi, theta = self.euler_angles
        z = self.length * np.sin(theta)
        x = self.length * np.cos(theta) * np.cos(phi)
        y = self.length * np.cos(theta) * np.sin(phi)
        return np.array([x, y, z])

    def update(self, dt):
        # Update the euler_angles and velocity of the pendlulum
        # theta_dotdot = -g / length * sin(theta)
        # phi_dotdot sin^2(theta) = 0
        length = self.length
        phi, theta = self.euler_angles
        phi_dot, theta_dot = self.angular_velocity

        phi = phi + phi_dot * dt / 2
        theta = theta + theta_dot * dt / 2
        phi_dot = phi_dot
        theta_dot = theta_dot - g / length * np.sin(theta) * dt
        phi = phi + phi_dot * dt / 2
        theta = theta + theta_dot * dt / 2

        self.euler_angles = np.array([phi, theta])
        self.angular_velocity = np.array([phi_dot, theta_dot])


def main():
    bsr.clear_mesh_objects()

    pendulum = Pendulum(
        length=1,
    )
    pendulum_blender = PendulumBlender(
        location=pendulum.position, ball_radius=0.25, cylinder_radius=0.03
    )

    dt = 10 ** (-2)
    framerate = 25
    simulation_ratio = int(1 / framerate / dt)
    time = np.arange(0, 100, dt)

    for k, t in enumerate(time):
        pendulum.update(dt)
        if k % simulation_ratio == 0:
            # Update the location of the pendulum
            pendulum_blender.update(position=pendulum.position)
            pendulum_blender.set_keyframe(keyframe=k)

    ### Saving the file ###
    write_filepath = "pendulum.blend"
    bsr.save(write_filepath)


if __name__ == "__main__":
    main()
