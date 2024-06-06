import matplotlib.pyplot as plt
import numpy as np

# Constants
v0 = 25  # initial velocity
g = 9.81  # gravitational acceleration, m/s^2
k = 0.1  # spring constant, N/m
dt = 0.001  # time step
b = 0.0  # air resistance coefficient

# Time array
time = np.arange(0, 5, dt)

# Analytical solution
y_analytical = v0 * time - 0.5 * g * time**2

# Numerical solution (Euler method)
y_numerical = np.zeros(time.shape)
v = v0
for i in range(1, len(time)):
    y = y_numerical[i - 1]
    y_numerical[i] = y + v * dt
    v -= g * dt + k * y * dt + b * v * dt

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(time, y_analytical, "r-", label="Analytical (Formula)")
plt.plot(time, y_numerical, "b--", label="Numerical (Euler)")
plt.xlabel("Time (s)")
plt.ylabel("Height (m)")
plt.title("Projectile Motion: Analytical vs. Numerical")
plt.legend()
plt.grid(True)
plt.show()

# Calculate L1 and L2 errors
l1_error = np.sum(np.abs(y_analytical - y_numerical)) / len(time)
l2_error = np.sqrt(np.sum((y_analytical - y_numerical) ** 2) / len(time))

print(f"L1 Error: {l1_error:.3f}")
print(f"L2 Error: {l2_error:.3f}")
