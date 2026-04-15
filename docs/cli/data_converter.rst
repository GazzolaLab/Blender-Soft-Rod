CLI Commands
============

The project ships command-line tools for Blender export and the Virtual Field
runtime.

Elastica data converter
-----------------------

Convert simulation data exported from `PyElastica`_ into a `.blend` file.

.. _PyElastica: https://github.com/GazzolaLab/PyElastica

.. click:: elastica_blender.converter.npz2blend:main
   :prog: elastica-npz2blend
   :nested: full
