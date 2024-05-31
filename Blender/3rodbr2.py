import os

import bpy
import numpy as np

npz_file_path = "/Users/adityakorlahalli/Downloads/br2_data (1).npz"
npz_data = np.load(npz_file_path)


rod1_position = npz_data["position_rod_0"]
rod1_xpos = []
rod1_ypos = []
rod1_zpos = []

rod1_listx = []
rod1_listy = []
rod1_listz = []

rod2_position = npz_data["position_rod_1"]
rod2_xpos = []
rod2_ypos = []
rod2_zpos = []

rod2_listx = []
rod2_listy = []
rod2_listz = []

rod3_position = npz_data["position_rod_2"]
rod3_xpos = []
rod3_ypos = []
rod3_zpos = []

rod3_listx = []
rod3_listy = []
rod3_listz = []

# remember to initialize to zeros at first

for i in range(len(rod1_position)):
    rod1_xpos = []
    rod1_ypos = []
    rod1_zpos = []

    rod2_xpos = []
    rod2_ypos = []
    rod2_zpos = []

    rod3_xpos = []
    rod3_ypos = []
    rod3_zpos = []

    for j in range(len(rod1_position[i][0])):
        rod1_xpos.append(rod1_position[i][0][j])
        rod1_ypos.append(rod1_position[i][1][j])
        rod1_zpos.append(rod1_position[i][2][j])

        rod2_xpos.append(rod2_position[i][0][j])
        rod2_ypos.append(rod2_position[i][1][j])
        rod2_zpos.append(rod2_position[i][2][j])

        rod3_xpos.append(rod3_position[i][0][j])
        rod3_ypos.append(rod3_position[i][1][j])
        rod3_zpos.append(rod3_position[i][2][j])

    rod1_listx.append(rod1_xpos)
    rod1_listy.append(rod1_ypos)
    rod1_listz.append(rod1_zpos)

    rod2_listx.append(rod2_xpos)
    rod2_listy.append(rod2_ypos)
    rod2_listz.append(rod2_zpos)

    rod3_listx.append(rod3_xpos)
    rod3_listy.append(rod3_ypos)
    rod3_listz.append(rod3_zpos)


class Sphere:
    def __init__(self, location):
        self.obj = self.create_sphere(location)

    def create_sphere(self, location):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.005, location=location)
        return bpy.context.active_object

    def update_position(self, newx, newy, newz):
        self.obj.location.z = newz
        self.obj.location.y = newy
        self.obj.location.x = newx


class Cylinder:
    def __init__(self, pos1, pos2):
        self.obj = self.create_cylinder(pos1, pos2)

    def create_cylinder(self, pos1, pos2):
        depth, center, angles = self.calc_cyl_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.005, depth=1, location=center
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

# creates 3 list of spheres and 3 lists of cylinders


# creates spheres
spheres1 = []
spheres2 = []
spheres3 = []

for i in range(len(rod1_listx[0])):
    spheres1.append(
        Sphere((rod1_listx[0][i], rod1_listy[0][i], rod1_listz[0][i]))
    )
    spheres2.append(
        Sphere((rod2_listx[0][i], rod2_listy[0][i], rod2_listz[0][i]))
    )
    spheres3.append(
        Sphere((rod3_listx[0][i], rod3_listy[0][i], rod3_listz[0][i]))
    )

cylinders1 = []
cylinders2 = []
cylinders3 = []
for i in range(len(spheres1) - 1):
    cylinders1.append(
        Cylinder(spheres1[i].obj.location, spheres1[i + 1].obj.location)
    )
    cylinders2.append(
        Cylinder(spheres2[i].obj.location, spheres2[i + 1].obj.location)
    )
    cylinders3.append(
        Cylinder(spheres3[i].obj.location, spheres3[i + 1].obj.location)
    )


# so this needs to be pretty reworked
for time_step in range(len(rod1_listx)):
    for s in range(len(spheres1)):
        spheres1[s].update_position(
            rod1_listx[time_step][s],
            rod1_listy[time_step][s],
            rod1_listz[time_step][s],
        )
        spheres1[s].obj.keyframe_insert(data_path="location", frame=time_step)
        spheres2[s].update_position(
            rod2_listx[time_step][s],
            rod2_listy[time_step][s],
            rod2_listz[time_step][s],
        )
        spheres2[s].obj.keyframe_insert(data_path="location", frame=time_step)
        spheres3[s].update_position(
            rod3_listx[time_step][s],
            rod3_listy[time_step][s],
            rod3_listz[time_step][s],
        )
        spheres3[s].obj.keyframe_insert(data_path="location", frame=time_step)

    for c in range(len(cylinders1)):
        depth, center, angles = cylinders1[c].calc_cyl_orientation(
            spheres1[c].obj.location, spheres1[c + 1].obj.location
        )
        cylinders1[c].obj.location = (center[0], center[1], center[2])
        cylinders1[c].obj.rotation_euler = (0, angles[1], angles[0])
        cylinders1[c].obj.scale[2] = depth
        # adding to keyframe
        cylinders1[c].obj.keyframe_insert(data_path="location", frame=time_step)
        cylinders1[c].obj.keyframe_insert(
            data_path="rotation_euler", frame=time_step
        )
        cylinders1[c].obj.keyframe_insert(data_path="scale", frame=time_step)

        depth, center, angles = cylinders2[c].calc_cyl_orientation(
            spheres2[c].obj.location, spheres2[c + 1].obj.location
        )
        cylinders2[c].obj.location = (center[0], center[1], center[2])
        cylinders2[c].obj.rotation_euler = (0, angles[1], angles[0])
        cylinders2[c].obj.scale[2] = depth
        # adding to keyframe
        cylinders2[c].obj.keyframe_insert(data_path="location", frame=time_step)
        cylinders2[c].obj.keyframe_insert(
            data_path="rotation_euler", frame=time_step
        )
        cylinders2[c].obj.keyframe_insert(data_path="scale", frame=time_step)

        depth, center, angles = cylinders3[c].calc_cyl_orientation(
            spheres3[c].obj.location, spheres3[c + 1].obj.location
        )
        cylinders3[c].obj.location = (center[0], center[1], center[2])
        cylinders3[c].obj.rotation_euler = (0, angles[1], angles[0])
        cylinders3[c].obj.scale[2] = depth
        # adding to keyframe
        cylinders3[c].obj.keyframe_insert(data_path="location", frame=time_step)
        cylinders3[c].obj.keyframe_insert(
            data_path="rotation_euler", frame=time_step
        )
        cylinders3[c].obj.keyframe_insert(data_path="scale", frame=time_step)
