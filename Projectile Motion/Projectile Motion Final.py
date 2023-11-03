#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import matplotlib.pyplot as plt

v0 = 25
x = np.array([0, v0])

g = 9.81

def f(x):
    y, v = x[0], x[1]
    return np.array([v, -g])

time = np.arange(0, 5, 10**(-3))

pos = [x[0]]
for k, t in enumerate(time[:-1]):
    x = x + f(x) * 10**(-3)
    pos.append(x[0])

#print(time, pos)

plt.plot(time, pos, '-b', label='Position (y) part a')
plt.xlabel('Time (t)')
plt.ylabel('Position (y)')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




