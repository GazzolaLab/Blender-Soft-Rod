Integration with Elastica
=========================

`PyElastica`_ is an *open-source* project for simulating suite of soft, slender, one-dimensional structures using Cosserat Rod theory.

Convert .npz to .blend
----------------------

Typically, simulation from `Pyelastica`_ is saved in dictionary format with the key `position_collection` and `radius_collection`.
The following example demonstrates the case where the simulation data contains 5 rods, 100 time steps, and 10 nodes per rod.

.. code-block:: python

    >> path = "data.npz"
    >> data = np.load(path)
    >> time = data['time']
    >> time.shape
    (100,)
    >> position_collection = data['position_collection']
    >> position_collection.shape
    (5, 100, 3, 10)
    >> radius_collection = data['radius_collection']
    >> radius_collection.shape
    (5, 100, 10)

To create a Blender animation from the simulation data, use the following code:

.. code-block:: bash

    elastica-npz2blender -p data.npz -o simulation.blend

The above command will create a Blender file `simulation.blend` with the animation of the rods.
The command line options are described in the

:doc:`../cli/data_converter`.

For multiple rod groups, one can include a `tag` option to distinguish between different rod groups.

.. code-block:: python

    >> path = "data.npz"
    >> data = np.load(path)
    >> time = data['time']
    >> time.shape
    (100,)
    >> data['straight_position_collection'].shape
    (5, 100, 3, 10)
    >> data['helical_position_collection'].shape
    (3, 100, 3, 12)

The corresponding command line option is:

.. code-block:: bash

    elastica-npz2blender -p data.npz -o simulation.blend --tag straight --tag helical


Using Callback
--------------

During the pyelastica simulation, one can use the `BlenderRodCallback` to create a Blender animation.
The following example is borrowed from `ButterflyCase`_ in the `PyElastica`_ repository.

.. code-block:: python

    import bsr
    from elastica_blender import BlenderRodCallback

    ...

    butterfly_sim.collect_diagnostics(butterfly_rod).using(
        BlenderRodCallback, step_skip=100
    )
    butterfly_sim.finalize()
    ...

    ea.integrate(timestepper, butterfly_sim, final_time, total_steps)

    bsr.save("butterfly.blend")  # Save the Blender file

The `BlenderRodCallback` will save the simulation visualization in the `butterfly.blend` file.


.. _PyElastica: https://github.com/GazzolaLab/PyElastica
.. _ButterflyCase: https://github.com/GazzolaLab/PyElastica/blob/master/examples/ButterflyCase/butterfly.py
