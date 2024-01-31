#Helper fn
#Call in main function; Replace vals in euler rotation and location for cylinders
#POTENTIALLY USE CYLINDER.SCALE[2] for length stretch on z-axis

#Spheres[i].location and spheres[i+1].location = pos1, pos2 pass in


import bpy
import numpy as np
    
    #Can access values one at a time if you return as an array

def calc_cyl_orientation(pos1,pos2):
    depth = np.linalg.norm(pos2 - pos1)
    dz =  pos2[2] - pos1[2]
    dy = pos2[1] - pos1[1]
    dx = pos2[0] - pos1[0]
    center = (pos1 + pos2) / 2
    # Spherical coords (phi and theta); Can be used in cyl. euler rotations (Look at wikipedia diagram)
    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz/depth)
    angles = np.array([phi,theta])
    #Should it return something?
    return depth, center, angles

def f(x):
    k = 5
    y, v = x[0], x[1]
    return np.array([v, (-1) * (k) * (y)])


# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

# Vector of all the values for initial velocities
v0_values = [20, 25, 30, 35, 40]

dt = 10**(-6)
framerate = 25
simulation_ratio = int(1 / framerate / dt)

time = np.arange(0, 10, dt)

# Create spheres with different initial velocities
spheres = []
for i, v0 in enumerate(v0_values):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(i * 2, 0, 0))
    sphere = bpy.context.active_object
    spheres.append((sphere, v0))

# Create cylinders to connect the spheres
cylinders = []
for i in range(len(spheres) - 1):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=2.5)
    cylinder = bpy.context.active_object
    cylinders.append(cylinder)


#The actual animation
for i, (sphere, v0) in enumerate(spheres):
    pos = [0]
    x = np.array([0, v0])
    for k, t in enumerate(time[:-1]):
        x = x + f(x) * dt

        pos.append(x[0])
        if (k % simulation_ratio) == 0:
            sphere.location.z = pos[k]
            sphere.keyframe_insert(data_path="location", index=2, frame=int(k/simulation_ratio) + 1)
            sphere.keyframe_insert(data_path="location", index=1, frame=int(k/simulation_ratio) + 1)
            sphere.keyframe_insert(data_path="location", index=0, frame=int(k/simulation_ratio) + 1)
           
            # Check if the current index is within the range of cylinders
            if i < len(cylinders):
                # Update the cylinder's location and rotation to connect the current sphere to the next one
                # xloc = sphere.location.x + (spheres[i+1][0].location.x - sphere.location.x) / 2
                depth, center, angles = calc_cyl_orientation(sphere.location, spheres[i+1][0].location)
                cylinders[i].location = (center[0], center[1], center[2])
                cylinders[i].rotation_euler = (angles[1], 0, angles[0])
               
               

                # Keyframe the cylinder's location and rotation
                cylinders[i].keyframe_insert(data_path="location", index=2, frame=int(k/simulation_ratio) + 1)
                cylinders[i].keyframe_insert(data_path="location", index=0, frame=int(k/simulation_ratio) + 1)
                cylinders[i].keyframe_insert(data_path="rotation_euler", index=2, frame=int(k/simulation_ratio) + 1)

