import bpy
import pytest


def test_import_bpy_singleton():
    import bpy as loaded_bpy

    assert (
        bpy == loaded_bpy
    ), f"bpy singleton not loaded correctly: {bpy} != {loaded_bpy}"

    loaded_bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2)
    sphere = bpy.context.active_object
    _sphere = loaded_bpy.context.active_object

    assert (
        sphere == _sphere
    ), f"bpy singleton not loaded correctly: {sphere} != {_sphere}"
    assert id(sphere) == id(
        _sphere
    ), f"bpy singleton not loaded correctly: {sphere} != {_sphere}"
