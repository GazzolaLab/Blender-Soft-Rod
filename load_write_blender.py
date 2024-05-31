import os
import numpy as np
import bpy

import bsr

class PendulumBlender:
    def __init__(self, location, ball_radius):
        self.location = location
        self.ball_radius = ball_radius

        origin_location = np.array([0, 0, 0])
        # Add a sphere
        self.add_sphere(
            location=self.location, 
            radius=self.ball_radius
        )
        # Add a cylinder
        center, angle, length = PendulumBlender.calculate_cylinder_pose(
            loc1=origin_location, 
            loc2=self.location
        )
        self.add_cylinder(
            location=center, 
            rotation=angle, 
            depth=length, 
            radius=0.1*self.ball_radius
        )

    def __repr__(self):
        return f"Pendulum at {self.location} with ball radius {self.ball_radius}"
    
    def add_sphere(self, location, radius):
        self.sphere = bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, 
            location=location
        )
        self.sphere = bpy.context.active_object

    def add_cylinder(self, location, rotation, depth, radius):
        bpy.ops.mesh.primitive_cylinder_add(
            location=location,
            depth=depth,
            radius=radius,
        )
        self.cylinder = bpy.context.active_object
        self.cylinder.rotation_euler = rotation

    @staticmethod
    def calculate_cylinder_pose(
        loc1: np.ndarray, 
        loc2: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # Calculate the depth of the cylinder
        length = np.linalg.norm(loc1 - loc2)
        # Calculate the center of the cylinder
        center = (loc1 + loc2) / 2
        # Calculate the difference in z, y, and x
        delta = loc2 - loc1
        # Calculate the angles of the cylinder
        angle = np.array([
            0, 
            np.arccos(delta[2] / length), 
            np.arctan2(delta[1], delta[0])
        ])
        return center, angle, length
    
    def update(self, position):
        self.sphere.location = position
        center, angle, length = PendulumBlender.calculate_cylinder_pose(
            loc1=np.array([0, 0, 0]), 
            loc2=position
        )
        self.cylinder.location = center
        self.cylinder.rotation_euler = angle
        self.cylinder.scale[2] = length

class Pendulum:
    def __init__(self, length, euler_angles):
        self.length = length
        self.euler_angles = euler_angles
        self.position = self.calculate_position()
        self.velocity = np.array([0., 0., 0.])
    
    def calculate_position(self):
        x = self.length * np.cos(self.euler_angles[1]) * np.cos(self.euler_angles[0])
        y = self.length * np.cos(self.euler_angles[1]) * np.sin(self.euler_angles[0])
        z = self.length * np.sin(self.euler_angles[1])
        return np.array([x, y, z])
    
    def get_position(self):
        return self.position
    
    def update(self, dt):
        # Update the velocity
        self.position += self.velocity * dt/2
        self.velocity += np.array([0, 0, -9.81]) * dt
        self.position += self.velocity * dt/2


def delete_all():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def main():
    delete_all()
    
    pendulum_length = 0.3
    pendulum_euler_angles = np.array([0., 0.])
    pendulum = Pendulum(
        length=pendulum_length, 
        euler_angles=pendulum_euler_angles
    )
    pendulum_blender = PendulumBlender(
        location=pendulum.get_position(), 
        ball_radius=0.2
    )
    
    dt = 10**(-3)
    framerate = 25
    simulation_ratio = int(1 / framerate / dt)
    time = np.arange(0, 10, dt)

    for k, t in enumerate(time):
        pendulum.update(dt)
        if k % simulation_ratio == 0:
            # Update the location of the pendulum
            pendulum_blender.update(position=pendulum.get_position())
            

            # # Update the scene
            bpy.context.view_layer.update()
    
    ### Saving the file ###
    write_filepath = "pendulum.blend"
    write_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), write_filepath)
    #Saves above script as .blend file; stores on personal device using filepath
    bpy.ops.wm.save_as_mainfile(filepath=write_filepath)  
    with bpy.data.libraries.load(load_filepath, link=True) as (
        data_from,
        data_to,
    ):
        data_to.objects = [
            name for name in data_from.objects if name.startswith("S")
        ]


    # load_filepath = "Blender/3rodbr2.blend"

    # with bpy.data.libraries.load(load_filepath, link=True) as (data_from, data_to):
    #     data_to.objects = [name for name in data_from.objects if name.startswith("S")]

    # for obj in data_to.objects:
    #     assert obj is not None
    #     print(obj.name)

    # # TODO: The writing part is not working
    # # data_to.objects["Cube"].select_set(True)
    # write_filepath = "/Blender/3rodbr2_write.blend"
    # bpy.data.libraries.write(write_filepath, set(bpy.context.selected_objects), path_remap="RELATIVE")

    
    
    # TODO: The writing part is not working
    # data_to.objects["Cube"].select_set(True)
    write_filepath = "/Blender/3rodbr2_write.blend"
    bpy.data.libraries.write(
        write_filepath, set(bpy.context.selected_objects), path_remap="RELATIVE"
    )


if __name__ == "__main__":
    main()
