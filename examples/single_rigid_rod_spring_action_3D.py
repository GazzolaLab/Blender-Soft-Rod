import numpy as np

import bsr

bsr.clear_mesh_objects()

# Environment configuration
k_x = 1  # spring constant
k_y = 3  # spring constant
k_z = 5  # spring constant
k = np.array([k_x, k_y, k_z])


def f(x):
    y, v = x[0], x[1]
    return np.array([v, (-1) * (k) * (y)])


def euler_forward_step(x, f, dt):
    return x + f(x) * dt


# Rod configuration
N = 5
velocity = np.stack(  # initial velocities
    [
        np.arange(0, 25, N),
        np.arange(10, 35, N),
        np.arange(20, 45, N),
    ],
    axis=0,
)  # (3, 5)
position = np.stack(
    [
        np.zeros(N),  # x
        np.zeros(N),  # y
        np.linspace(0, 10, N),  # z
    ],
    axis=0,
)  # (3, 5)
radius = 0.2 * np.ones(N)  # radius of the rod

rod = bsr.rod()
rod.update(keyframe=0, positions=position, radius=radius)


####### SIMULATION ########
# Simulation parameters

dt = 10 ** (-3)
framerate = 25
simulation_ratio = int(1 / framerate / dt)
time = np.arange(0, 10, dt)

# Euler-Forward time stepper
for time_index, t in enumerate(time[:-1]):
    state = np.stack([position, velocity], axis=0)
    position[:] = euler_forward_step(state, f, dt)

    if time_index % simulation_ratio == 0:
        # update the rod
        keyframe = int(time_index / simulation_ratio) + 1
        rod.update(keyframe=keyframe, positions=position, radius=radius)

bsr.save("single_rigid_rod_spring_action_3D.blend")
