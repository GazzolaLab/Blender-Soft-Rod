import matplotlib.pyplot as plt
import numpy as np

import bsr

bsr.clear_mesh_objects()

# Environment configuration
k = 5.0  # spring constant
m = 1.0  # mass


def f(x):
    y, v = x[0], x[1]
    return np.array([v, -(k / m) * y])


def euler_forward_step(x, f, dt):
    return x + f(x) * dt


def analytical_solution(t, y0: float, v0: float):
    omega = np.sqrt(k)
    return np.cos(omega * t) * y0 + np.sin(omega * t) * v0 / omega


# Rod configuration
N = 5
velocity = np.linspace(0, 1, N)  # initial z-velocities
positions = np.stack(
    [
        np.zeros(N),  # x
        np.zeros(N),  # y
        np.linspace(0, 3, N),  # z
    ],
    axis=0,
)  # (3, 5)
radii = 0.2 * np.ones(N - 1)  # radius of the rod

rod = bsr.Rod(positions=positions, radii=radii)


####### SIMULATION ########
# Simulation parameters

dt = 10 ** (-3)
framerate = 25
simulation_ratio = int(1 / framerate / dt)
time = np.arange(0, 10, dt)
analytical_solution = analytical_solution(time, positions[1][-1], velocity[-1])
simulated_solution = np.zeros(len(time))

# Euler-Forward time stepper
for time_index, t in enumerate(time):
    state = np.stack([positions[1], velocity], axis=0)
    next_state = euler_forward_step(state, f, dt)
    positions[1], velocity = next_state[0], next_state[1]
    simulated_solution[time_index] = positions[1][-1]

    if time_index % simulation_ratio == 0:
        # update the rod
        keyframe = int(time_index / simulation_ratio) + 1
        rod.update_states(positions=positions, radii=radii)
        rod.update_keyframe(keyframe)

bsr.save("single_rigid_rod_spring_action.blend")

# Plot L1 and L2 error
l1_error = np.abs(analytical_solution - simulated_solution)
l2_error = np.sqrt(np.square(analytical_solution - simulated_solution))
plt.plot(time, l1_error, label="L1 error")
plt.plot(time, l2_error, label="L2 error")
plt.xlabel("time")
plt.ylabel("Tip Error")
plt.legend()
plt.grid(True)
plt.savefig("single_rigid_rod_spring_action_error.png", dpi=300)
