import matplotlib.pyplot as plt
import numpy as np

import bsr
from bsr.geometry import Sphere

bsr.clear_mesh_objects()

# Constants
v0 = 25  # initial velocity
g = 9.81  # gravitational acceleration, m/s^2
k = 0.1  # spring constant, N/m
dt = 0.001  # time step
b = 0.0  # air resistance coefficient

# Time array
time = np.arange(0, 5, dt)

# Blender object
sphere = Sphere(radius=0.5, position=np.array((0, 0, 0)))

# Analytical solution
y_analytical = v0 * time - 0.5 * g * time**2

# Numerical solution (Euler method)
y_numerical = np.zeros(time.shape)
v = v0
for i in range(1, len(time)):
    y = y_numerical[i - 1]
    y_numerical[i] = y + v * dt
    v -= g * dt + k * y * dt + b * v * dt

    # Update sphere position
    sphere.update_states(position=np.array([0, 0, y]))
    sphere.set_keyframe(i)

# Blender file save
bsr.save("projectile_motion.blend")

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(time, y_analytical, "r-", label="Analytical (Formula)")
plt.plot(time, y_numerical, "b--", label="Numerical (Euler)")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Projectile Motion: Analytical vs. Numerical")
plt.legend()
plt.grid(True)
plt.savefig("projectile_motion.png")
plt.close("all")

# Calculate L1 and L2 errors
l1_error = np.sum(np.abs(y_analytical - y_numerical)) / len(time)
l2_error = np.sqrt(np.sum((y_analytical - y_numerical) ** 2) / len(time))

# Print L1 and L2 convergence
plt.figure(figsize=(10, 5))
plt.plot(time, np.abs(y_analytical - y_numerical), "g-", label="Error")
plt.xlabel("Time (s)")
plt.ylabel("Error (m)")
plt.title("Projectile Motion: Error Convergence")
plt.legend()
plt.grid(True)
plt.savefig("projectile_motion_error.png")
plt.close("all")
