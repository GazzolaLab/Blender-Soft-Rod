# Originally from PyElastica example case

import elastica as ea
import numpy as np

import bsr
from elastica_blender import BlenderRodCallback


class TimoshenkoBeamSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Forcing,
    ea.CallBacks,
    ea.Damping,
):
    pass


timoshenko_sim = TimoshenkoBeamSimulator()
final_time = 5000.0

# setting up test params
n_elem = 100
start = np.zeros((3,))
direction = np.array([0.0, 0.0, 1.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 3.0
base_radius = 0.25
base_area = np.pi * base_radius**2
density = 5000
nu = 0.1 / 7 / density / base_area
E = 1e6
# For shear modulus of 1e4, nu is 99!
poisson_ratio = 99
shear_modulus = E / (poisson_ratio + 1.0)

shearable_rod = ea.CosseratRod.straight_rod(
    n_elem,
    start,
    direction,
    normal,
    base_length,
    base_radius,
    density,
    youngs_modulus=E,
    shear_modulus=shear_modulus,
)

timoshenko_sim.append(shearable_rod)
# add damping
dl = base_length / n_elem
dt = 0.07 * dl
timoshenko_sim.dampen(shearable_rod).using(
    ea.AnalyticalLinearDamper,
    damping_constant=nu,
    time_step=dt,
)

timoshenko_sim.constrain(shearable_rod).using(
    ea.OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)

end_force = np.array([-15.0, 0.0, 0.0])
timoshenko_sim.add_forcing_to(shearable_rod).using(
    ea.EndpointForces, 0.0 * end_force, end_force, ramp_up_time=final_time / 2.0
)

# Use elastica_blender.BlenderRodCallback to render rod-video
timoshenko_sim.collect_diagnostics(shearable_rod).using(
    BlenderRodCallback,
    step_skip=500,
)

timoshenko_sim.finalize()
timestepper = ea.PositionVerlet()

total_steps = int(final_time / dt)
ea.integrate(timestepper, timoshenko_sim, final_time, total_steps)

# Save as .blend file
bsr.save("timoshenko_beam.blend")
