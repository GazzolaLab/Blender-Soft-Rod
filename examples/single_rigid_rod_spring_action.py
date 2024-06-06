import matplotlib.pyplot as plt
import numpy as np

import bsr

bsr.clear_mesh_objects()

# Environment configuration
k = 5  # spring constant


def f(x):
    y, v = x[0], x[1]
    return np.array([v, (-1) * (k) * (y)])


def euler_forward_step(x, f, dt):
    return x + f(x) * dt


def analytical_solution(t, y0, v0):
    omega = np.sqrt(k)
    return y0 * np.cos(omega * t) + v0 / omega * np.sin(omega * t)


# Rod configuration
N = 5
velocity = np.arange(20, 45, N)  # initial z-velocities
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
analytical_solution = analytical_solution(time, position[2], velocity)

# Euler-Forward time stepper
for time_index, t in enumerate(time[:-1]):
    state = np.stack([position[2], velocity], axis=0)
    position[2] = euler_forward_step(state, f, dt)

    if time_index % simulation_ratio == 0:
        # update the rod
        keyframe = int(time_index / simulation_ratio) + 1
        rod.update(keyframe=keyframe, positions=position, radius=radius)

bsr.save("single_rigid_rod_spring_action.blend")

# Plot L1 and L2 error
l1_error = np.abs(analytical_solution - position[2]).sum(axis=0)
l2_error = np.sqrt(np.square(analytical_solution - position[2]).sum(axis=0))
plt.plot(velocity, l1_error, label="L1 error")
plt.plot(velocity, l2_error, label="L2 error")
plt.xlabel("Initial velocity")
plt.ylabel("Error")
plt.legend()
plt.grid(True)
plt.savefig("single_rigid_rod_spring_action_error.png", dpi=300)
