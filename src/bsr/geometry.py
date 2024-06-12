import colorsys

import bpy
import numpy as np


class Sphere:
    def __init__(self, location, radius=0.005):
        self.obj = self.create_sphere(location, radius)

    def create_sphere(self, location, radius):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location)
        return bpy.context.active_object

    def update_position(self, location):
        self.obj.location.z = location[2]
        self.obj.location.y = location[1]
        self.obj.location.x = location[0]


class Cylinder:
    def __init__(self, pos1, pos2):
        self.obj = self.create_cylinder(pos1, pos2)
        self.mat = bpy.data.materials.new(name="cyl_mat")
        self.obj.active_material = self.mat

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

    def update_position(self, pos1, pos2):
        depth, center, angles = self.calc_cyl_orientation(pos1, pos2)
        self.obj.location = (center[0], center[1], center[2])
        self.obj.rotation_euler = (0, angles[1], angles[0])
        self.obj.scale[2] = depth

        # computing deformation heat-map
        max_def = 0.07

        h = (
            -np.sqrt(self.obj.location[0] ** 2 + self.obj.location[2] ** 2)
            / max_def
            + 240 / 360
        )
        v = (
            np.sqrt(self.obj.location[0] ** 2 + self.obj.location[2] ** 2)
            / max_def
            * 0.5
            + 0.5
        )

        r, g, b = colorsys.hsv_to_rgb(h, 1, v)
        self.update_color(r, g, b, 1)

    def update_color(self, r, g, b, a):
        self.mat.diffuse_color = (r, g, b, a)


# TODO: Refactor
class Frustum:
    def __init__(self, pos1, pos2, radius1, radius2):
        self.obj = self.create_frustum(pos1, pos2, radius1, radius2)
        self.mat = bpy.data.materials.new(name="cyl_mat")
        self.obj.active_material = self.mat

    def create_frustum(self, pos1, pos2, radius1, radius2):
        depth, center, angles = self.calc_frust_orientation(pos1, pos2)
        bpy.ops.mesh.primitive_cone_add(
            radius1=radius1, radius2=radius2, depth=1, location=center
        )
        frustum = bpy.context.active_object
        frustum.rotation_euler = (0, angles[1], angles[0])
        frustum.scale[2] = depth
        return frustum

    def calc_frust_orientation(self, pos1, pos2):
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

    def update_position(self, pos1, pos2):
        depth, center, angles = self.calc_frust_orientation(pos1, pos2)
        self.obj.location = (center[0], center[1], center[2])
        self.obj.rotation_euler = (0, angles[1], angles[0])
        self.obj.scale[2] = depth

        # computing deformation heat-map
        max_def = 0.07

        h = (
            -np.sqrt(self.obj.location[0] ** 2 + self.obj.location[2] ** 2)
            / max_def
            + 240 / 360
        )
        v = (
            np.sqrt(self.obj.location[0] ** 2 + self.obj.location[2] ** 2)
            / max_def
            * 0.5
            + 0.5
        )

        r, g, b = colorsys.hsv_to_rgb(h, 1, v)
        self.update_color(r, g, b, 1)

    def update_color(self, r, g, b, a):
        self.mat.diffuse_color = (r, g, b, a)

    def update(self, pos1, pos2, time_step):
        self.update_position(pos1, pos2)
        # adding to keyframe
        self.obj.keyframe_insert(data_path="location", frame=time_step)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=time_step)
        self.obj.keyframe_insert(data_path="scale", frame=time_step)
        self.mat.keyframe_insert(data_path="diffuse_color", frame=time_step)
