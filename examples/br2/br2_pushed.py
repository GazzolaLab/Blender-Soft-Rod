"""
Created on Jul 25, 2024
@author: Heng-Sheng (Hanson) Chang
"""

from abc import ABC, abstractmethod

import elastica as ea
import numpy as np
from callbacks import BlenderBR2CallBack, BR2Property
from tqdm import tqdm

import bsr


class BaseSimulator(
    ea.BaseSystemCollection,
    ea.Damping,
    ea.CallBacks,
    ea.Constraints,
    ea.Forcing,
):
    pass


class BaseEnvironment(ABC):
    def __init__(
        self,
        final_time: float,
        time_step: float = 1.0e-5,
        recording_fps: int = 30,
    ) -> None:
        self.StatefulStepper = ea.PositionVerlet()  # Integrator type

        self.final_time = final_time
        self.time_step = time_step
        self.total_steps = int(self.final_time / self.time_step)
        self.recording_fps = recording_fps
        self.step_skip = int(1.0 / (self.recording_fps * self.time_step))
        self.reset()

    def reset(
        self,
    ) -> None:
        # Initialize the simulator
        self.simulator = BaseSimulator()

        self.setup()

        # Finalize the simulator and create time stepper
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = ea.extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )

    def step(self, time: float) -> float:
        # Run the simulation for one step
        time = self.do_step(
            self.StatefulStepper,
            self.stages_and_updates,
            self.simulator,
            time,
            self.time_step,
        )

        # Return current simulation time
        return time

    @abstractmethod
    def setup(
        self,
    ) -> None:
        pass


class BR2Environment(BaseEnvironment):
    def __init__(self, *args, **kwargs) -> None:
        bsr.clear_mesh_objects()
        super().__init__(*args, **kwargs)

    def setup(
        self,
    ) -> None:
        self.setup_BR2()

    def setup_BR2(
        self,
    ) -> None:
        # BR2 arm parameters
        direction = np.array(
            [0.0, 0.0, -1.0]
        )  # direction of the BR2 arm (z-axis pointing down)
        normal = np.array(
            [1.0, 0.0, 0.0]
        )  # bending FREE direction of the BR2 arm (x-axis pointing forward)
        n_elements = 100  # number of discretized elements of the BR2 arm
        rest_length = 0.16  # rest length of the BR2 arm
        rest_radius = 0.015  # rest radius of the BR2 arm
        density = 700  # density of the BR2 arm
        youngs_modulus = 1e7  # Young's modulus of the BR2 arm
        poisson_ratio = 0.5  # Poisson's ratio of the BR2 arm
        damping_constant = 0.05  # damping constant of the BR2 arm

        # Setup a rod
        self.rod = ea.CosseratRod.straight_rod(
            n_elements=n_elements,
            start=np.zeros((3,)),
            direction=direction,
            normal=normal,
            base_length=rest_length,
            base_radius=rest_radius * np.ones(n_elements),
            density=density,
            youngs_modulus=youngs_modulus,
            shear_modulus=youngs_modulus / (poisson_ratio + 1.0),
        )
        self.simulator.append(self.rod)

        # Setup viscous damping
        self.simulator.dampen(self.rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        # Setup boundary conditions
        self.simulator.constrain(self.rod).using(
            ea.OneEndFixedBC,
            constrained_position_idx=(0,),
            constrained_director_idx=(0,),
        )

        # Setup gravity force
        acc_gravity = np.array([0.0, 0.0, -9.80665])
        self.simulator.add_forcing_to(self.rod).using(
            ea.GravityForces, acc_gravity=acc_gravity
        )

        end_force = np.array([-15.0, 0.0, 0.0])
        self.simulator.add_forcing_to(self.rod).using(
            ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=1.0
        )

        actuation_radius_ratio = np.sqrt(3) / (2 + np.sqrt(3))

        offset_position_ratio = 2 / (2 + np.sqrt(3))

        bending_actuation_direction = np.array([1.0, 0.0, 0.0])

        rotation_CW_actuation_rotate_angle = 120 / 180 * np.pi
        rotation_CW_actuation_direction = np.array(
            [
                np.cos(rotation_CW_actuation_rotate_angle),
                np.sin(rotation_CW_actuation_rotate_angle),
                0.0,
            ]
        )

        rotation_CCW_actuation_rotate_angle = 240 / 180 * np.pi
        rotation_CCW_actuation_direction = np.array(
            [
                np.cos(rotation_CCW_actuation_rotate_angle),
                np.sin(rotation_CCW_actuation_rotate_angle),
                0.0,
            ]
        )

        br2_property = BR2Property(
            radii=(
                actuation_radius_ratio * rest_radius * np.ones(n_elements - 1)
            ),
            bending_actuation_position=np.tile(
                rest_radius
                * offset_position_ratio
                * bending_actuation_direction,
                (n_elements, 1),
            ).T,
            rotation_CW_actuation_position=np.tile(
                rest_radius
                * offset_position_ratio
                * rotation_CW_actuation_direction,
                (n_elements, 1),
            ).T,
            rotation_CCW_actuation_position=np.tile(
                rest_radius
                * offset_position_ratio
                * rotation_CCW_actuation_direction,
                (n_elements, 1),
            ).T,
        )

        # Setup blender rod callback
        self.simulator.collect_diagnostics(self.rod).using(
            BlenderBR2CallBack,
            step_skip=self.step_skip,
            property=br2_property,
            system=self.rod,
        )

    def save(self, filename: str) -> None:
        if filename.endswith(".blend"):
            filename = filename[:-6]

        # Save as .blend file
        bsr.save(filename + ".blend")


def main(
    final_time: float = 3.0,
    time_step: float = 1.0e-5,
    recording_fps: int = 30,
):
    # Initialize the environment
    env = BR2Environment(
        final_time=final_time, time_step=time_step, recording_fps=recording_fps
    )

    # Start the simulation
    print("Running simulation ...")
    time = np.float64(0.0)
    for step in tqdm(range(env.total_steps)):
        time = env.step(time=time)
    print("Simulation finished!")

    # Save the simulation
    env.save("BR2_simulation")


if __name__ == "__main__":
    main()
