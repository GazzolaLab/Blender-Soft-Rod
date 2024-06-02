import numpy as np

import bsr


def calc_cyl_orientation(pos1, pos2):
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    depth = np.linalg.norm(pos2 - pos1)
    dz = pos2[2] - pos1[2]
    dy = pos2[1] - pos1[1]
    dx = pos2[0] - pos1[0]
    center = (pos1 + pos2) / 2
    # Spherical coords (phi and theta); Can be used in cyl. euler rotations (Look at wikipedia diagram)
    phi = np.arctan2(dy, dx)
    theta = np.arccos(dz / depth)
    angles = np.array([phi, theta])
    # Should it return something?
    return depth, center, angles


def f(x):
    k = 5
    y, v = x[0], x[1]
    return np.array([v, (-1) * (k) * (y)])


# Clear existing mesh objects in the scene
bpy.ops.object.select_all(action="DESELECT")
bpy.ops.object.select_by_type(type="MESH")
bpy.ops.object.delete()


# Create spheres with different initial velocities

v0_values = [20, 25, 30, 35, 40]
spheres = (
    []
)  # sphere is now a list a tuples containing the sphere object and its z-velocity
for i, v0 in enumerate(v0_values):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(i * 2, 0, 0))
    sphere = bpy.context.active_object
    spheres.append([sphere, v0])

# Create cylinders to connect the spheres
cylinders = []
for i in range(len(spheres) - 1):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.1, depth=1)
    cylinder = bpy.context.active_object

    # connect cylinders to spheres on creation
    depth, center, angles = calc_cyl_orientation(
        spheres[i][0].location, spheres[i + 1][0].location
    )
    cylinder.location = (center[0], center[1], center[2])
    cylinder.rotation_euler = (0, angles[1], angles[0])
    cylinder.scale[2] = depth
    cylinders.append(cylinder)


####### SIMULATION ########


# Creating Simulation Parameters

# dt = 10**(-6)
dt = 10 ** (-3)
framerate = 25
simulation_ratio = int(1 / framerate / dt)
time = np.arange(0, 10, dt)

# Simulate the behavior of all objects over "time"
for time_index, t in enumerate(time[:-1]):
    for i, [sphere, vz] in enumerate(spheres):
        # update 3D-world position of each sphere
        x = np.array([sphere.location.z, vz])
        x = x + f(x) * dt
        sphere.location.z = x[0]
        sphere.location.y = x[0]
        spheres[i][1] = x[1]

    if (
        time_index % simulation_ratio
    ) == 0:  # this is an index which we want to write to a keyframe
        # then, we add updated sphere locations to the keyframe
        for i, (sphere, vz) in enumerate(spheres):
            sphere.keyframe_insert(
                data_path="location",
                frame=int(time_index / simulation_ratio) + 1,
            )

        # now we update cylinder orientation and then draw those to the keyframe.
        for i in range(len(cylinders)):
            depth, center, angles = calc_cyl_orientation(
                spheres[i][0].location, spheres[i + 1][0].location
            )
            cylinders[i].location = (center[0], center[1], center[2])
            cylinders[i].rotation_euler = (0, angles[1], angles[0])
            cylinders[i].scale[2] = depth

            # Keyframe the cylinder's location , rotation, and scaling
            cylinders[i].keyframe_insert(
                data_path="location",
                frame=int(time_index / simulation_ratio) + 1,
            )
            cylinders[i].keyframe_insert(
                data_path="rotation_euler",
                frame=int(time_index / simulation_ratio) + 1,
            )
            cylinders[i].keyframe_insert(
                data_path="scale", frame=int(time_index / simulation_ratio) + 1
            )
