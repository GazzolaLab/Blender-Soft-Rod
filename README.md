# Blender-Soft-Arm-Simulation

There are two main folders in this repository:
  - Projectile Motion
  - Blender

# Projectile Motion
  - Projectile Motion: graphs the movement of an object with the projectile motion formula
  - Projectile Motion Final: Plotting projectile motion used discretized methods
  - Proejctile Motion (-ky) as gravity: Replaces gravity with a spring constant to create conservative oscillatory motion
  - Projectile Motion (-ky - bv): Replaces gravity with -ky -bv to create damped oscialltory motion 

# Blender
  - rod_sim.py: The python script for a soft arm (spherical joints which are connected by rod linkages) performing oscillatory motion. The lengths of the linkages change as the joints seperate from each other, and the arm maintains its shape through the stretch.
  - Rod_simulation.blender: The blender animation for the arm performing the oscillatory motion.
