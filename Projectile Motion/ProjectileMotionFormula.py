#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

import matplotlib.pyplot as plt

 

def initVelocity(v):

    return v

 

v = initVelocity(25)

 

time = np.arange(0, 5, 10**(-3))

 

y = np.zeros(time.shape)

 

for k, t in enumerate(time):

    y[k] = v * t - 0.5 * 9.8 * t**2

   

plt.plot(time, y, '-b', label = 'Position (y) part a')

plt.xlabel('Time (t)')

plt.ylabel('Position (y)')

plt.legend()

plt.grid(True)

plt.show()




# In[ ]:




