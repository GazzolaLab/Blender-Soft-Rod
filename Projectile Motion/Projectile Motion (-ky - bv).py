#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt

v0 = 25
x = np.array([0, v0])
dt = 10**(-3)
 
def f(x):
    k = 5
    b = 1
    y, v= x[0], x[1]
    return np.array([v, ((-1) * k * y) + ((-1) * b * v)])

time = np.arange(0, 30, dt)
pos = [x[0]]
for k, t in enumerate(time[:-1]):
    x = x + f(x) * dt
    pos.append(x[0])

plt.plot(time, pos, '-b', label='Position (y) part a')
plt.xlabel('Time (t)')
plt.ylabel('Position (y)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




