"""
#Helper fn
#Call in main function; Replace vals in euler rotation and location for cylinders
#POTENTIALLY USE CYLINDER.SCALE[2] for length stretch on z-axis

#Spheres[i].location and spheres[i+1].location = pos1, pos2 pass in


import bpy
import numpy as np

    #Can access values one at a time if you return as an array

def calc_cyl_orientation(pos1,pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    depth = np.linalg.norm(pos2 - pos1)
    dz =  pos2[2] - pos1[2]
    dy = pos2[1] - pos1[1]
    dx = pos2[0] - pos1[0]
    center = (pos1 + pos2) / 2
    # Spherical coords (phi and theta); Can be used in cyl. euler rotations (Look at wikipedia diagram)
    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz/depth)
    angles = np.array([phi,theta])
    return depth, center, angles

#def f(x):
#    k = 5
#    y, v = x[0], x[1]
#    return np.array([v, (-1) * (k) * (y)])

spring_const1 = 5
spring_const2 = 3
def increment_position_z(pos_and_vel):
    z_position, velocity = pos_and_vel[0], pos_and_vel[1]
    return np.array([velocity, (-1) * spring_const1 * z_position])

def increment_position_y(pos_and_vel):
    y_position, velocity = pos_and_vel[0], pos_and_vel[1]
    return np.array([velocity, (-1) * spring_const2 * y_position])

# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()


# Create spheres with different initial velocities

v01_values = [20, 25, 30, 35, 40]

v02_values = [20, 25, 30, 35, 40]
spheres = [] # sphere is now a list a tuples containing the sphere object and its z-velocity
for i in range(len(v01_values)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=.2, location=(i * 2, 0, 0))
    sphere = bpy.context.active_object
#    spheres.append([sphere, v0])
    spheres.append([sphere, v01_values[i], v02_values[i]])

# Create cylinders to connect the spheres
cylinders = []
for i in range(len(spheres) - 1):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1)
    cylinder = bpy.context.active_object

    #connect cylinders to spheres on creation
    depth, center, angles = calc_cyl_orientation(spheres[i][0].location, spheres[i+1][0].location)
    cylinder.location = (center[0], center[1], center[2])
    cylinder.rotation_euler = (0,angles[1], angles[0])
    cylinder.scale[2] = depth
    cylinders.append(cylinder)


####### SIMULATION ########


# Creating Simulation Parameters

#dt = 10**(-6)
dt = 10**(-3)
framerate = 25
simulation_ratio = int(1 / framerate / dt)
time = np.arange(0, 10, dt)

# Simulate the behavior of all objects over "time"
for time_index, t in enumerate(time[:-1]):
    for i, [sphere, vel_z, vel_y] in enumerate(spheres):
        # update 3D-world position of each sphere
        zpos_and_vel = np.array([sphere.location.z, vel_z])
        zpos_and_vel = zpos_and_vel + increment_position_z(zpos_and_vel) * dt

        ypos_and_vel  = np.array([sphere.location.y, vel_y])
        ypos_and_vel = ypos_and_vel + increment_position_y(ypos_and_vel) * dt

        sphere.location.z = zpos_and_vel[0]
        sphere.location.y = ypos_and_vel[0]

        spheres[i][1] = zpos_and_vel[1]
        spheres[i][2] = ypos_and_vel[1]

    if (time_index % simulation_ratio) == 0: # this is an index which we want to write to a keyframe
        # then, we add updated sphere locations to the keyframe
        for i, (sphere, vel_z, vel_y) in enumerate(spheres):
            sphere.keyframe_insert(data_path="location", frame=int(time_index/simulation_ratio) + 1)

        #now we update cylinder orientation and then draw those to the keyframe.
        for i in range(len(cylinders)):
            depth, center, angles = calc_cyl_orientation(spheres[i][0].location, spheres[i+1][0].location)
            cylinders[i].location = (center[0], center[1], center[2])
            cylinders[i].rotation_euler = (0,angles[1], angles[0])
            cylinders[i].scale[2] = depth

            # Keyframe the cylinder's location , rotation, and scaling
            cylinders[i].keyframe_insert(data_path="location", frame=int(time_index/simulation_ratio) + 1)
            cylinders[i].keyframe_insert(data_path="rotation_euler", frame=int(time_index/simulation_ratio) + 1)
            cylinders[i].keyframe_insert(data_path="scale", frame=int(time_index/simulation_ratio) + 1)



"""

import bpy
import numpy as np


class Sphere:
    def __init__(self, location, vel_z, vel_y):
        self.obj = self.create_sphere(location)
        self.vel_z = vel_z
        self.vel_y = vel_y

    def create_sphere(self, location):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=location)
        return bpy.context.active_object

    def update_position(self, dt):
        zpos_and_vel = np.array([self.obj.location.z, self.vel_z])
        zpos_and_vel = (
            zpos_and_vel + self.increment_position_z(zpos_and_vel) * dt
        )
        ypos_and_vel = np.array([self.obj.location.y, self.vel_y])
        ypos_and_vel = (
            ypos_and_vel + self.increment_position_y(ypos_and_vel) * dt
        )
        self.obj.location.z = zpos_and_vel[0]
        self.obj.location.y = ypos_and_vel[0]
        self.vel_z = zpos_and_vel[1]
        self.vel_y = ypos_and_vel[1]

    def increment_position_z(self, pos_and_vel):
        spring_const1 = 5
        z_position, velocity = pos_and_vel[0], pos_and_vel[1]
        return np.array([velocity, (-1) * spring_const1 * z_position])

    def increment_position_y(self, pos_and_vel):
        spring_const2 = 3
        y_position, velocity = pos_and_vel[0], pos_and_vel[1]
        return np.array([velocity, (-1) * spring_const2 * y_position])


class Cylinder:
    def __init__(self, pos1, pos2):
        self.obj = self.create_cylinder(pos1, pos2)

    def create_cylinder(self, pos1, pos2):
        depth, center, angles = self.calc_cyl_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.1, depth=1, location=center
        )
        cylinder = bpy.context.active_object
        cylinder.rotation_euler = (0, angles[1], angles[0])
        cylinder.scale[2] = depth
        return cylinder

    def calc_cyl_orientation(self, pos1, pos2):
        pos1 = np.array(pos1)
        pos2 = np.array(pos2)
        depth = np.linalg.norm(pos2 - pos1)
        dz = pos2[2] - pos1[2]
        dy = pos2[1] - pos1[1]
        dx = pos2[0] - pos1[0]
        center = (pos1 + pos2) / 2
        phi = np.arctan2(dy, dx)
        theta = np.arccos(dz / depth)
        angles = np.array([phi, theta])
        return depth, center, angles


# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()

# Create spheres with different initial velocities
init_vel_z = [20, 25, 30, 35, 40]
init_vel_y = [20, 25, 30, 35, 40]

# creates spheres
spheres = [
    Sphere((i * 2, 0, 0), vel_z, vel_y)
    for i, (vel_z, vel_y) in enumerate(zip(init_vel_z, init_vel_y))
]


# Create cylinders to connect the spheres
cylinders = [
    Cylinder(spheres[i].obj.location, spheres[i + 1].obj.location)
    for i in range(len(spheres) - 1)
]

# SIMULATION
dt = 10 ** (-3)
framerate = 25
simulation_ratio = int(1 / framerate / dt)
time = np.arange(0, 10, dt)

for time_index, t in enumerate(time[:-1]):
    for sphere in spheres:
        sphere.update_position(dt)

    if (time_index % simulation_ratio) == 0:
        for sphere in spheres:
            sphere.obj.keyframe_insert(
                data_path="location",
                frame=int(time_index / simulation_ratio) + 1,
            )

        for i, cylinder in enumerate(cylinders):

            depth, center, angles = cylinder.calc_cyl_orientation(
                spheres[i].obj.location, spheres[i + 1].obj.location
            )
            cylinder.obj.location = (center[0], center[1], center[2])
            cylinder.obj.rotation_euler = (0, angles[1], angles[0])
            cylinder.obj.scale[2] = depth

            cylinder.obj.keyframe_insert(
                data_path="location",
                frame=int(time_index / simulation_ratio) + 1,
            )
            cylinder.obj.keyframe_insert(
                data_path="rotation_euler",
                frame=int(time_index / simulation_ratio) + 1,
            )
            cylinder.obj.keyframe_insert(
                data_path="scale", frame=int(time_index / simulation_ratio) + 1
            )
