import pathlib

from pathlib import Path
import bpy
from bsr.blender_commands.file import reload, save
import numpy as np
import pytest


# Test file creation
def test_file_create_using_bpy(tmp_path):
    blend_file = tmp_path / "test.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
    assert blend_file.exists()


@pytest.fixture(scope="function")
def blend_file(tmp_path_factory):
    blend_file_path = (
        tmp_path_factory.mktemp("data") / "test_file_handling.blend"
    )
    bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(1, 2, 3))
    obj = bpy.context.active_object
    obj.scale[0] = 0.1
    obj.scale.y = 0.2
    obj.scale.z = 0.3
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file_path))
    return blend_file_path


def test_file_opening_using_bpy(blend_file):
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    loaded_object = bpy.context.active_object
    np.testing.assert_allclose(loaded_object.scale.x, 0.1)
    np.testing.assert_allclose(loaded_object.scale.y, 0.2)
    np.testing.assert_allclose(loaded_object.scale.z, 0.3)

    loaded_object_location = loaded_object.location
    np.testing.assert_allclose(loaded_object_location.x, 1.0)
    np.testing.assert_allclose(loaded_object_location.y, 2.0)
    np.testing.assert_allclose(loaded_object_location.z, 3.0)


def test_file_opening_and_writing_data_using_bpy(blend_file):
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    new_radius = 0.2
    # - Note; cannot driectly change sphere radius post-definition; can work around by scaling it
    loaded_object = bpy.context.active_object

    scale_factor = 2
    loaded_object.scale = (
        scale_factor * loaded_object.scale[0],
        scale_factor * loaded_object.scale[1],
        scale_factor * loaded_object.scale[2],
    )
    # This effectively scales in each axis by the scale factor (2), increasing radius to 0.2

    loaded_object.location = (4, 5, 6)

    new_blend_file = blend_file.parent / "testname"
    bpy.ops.wm.save_as_mainfile(filepath=str(new_blend_file))
    assert new_blend_file.exists()

    bpy.ops.wm.open_mainfile(filepath=str(new_blend_file))
    loaded_object = bpy.context.active_object
    np.testing.assert_allclose(loaded_object.scale.x, 0.2, rtol=1e-6)
    np.testing.assert_allclose(loaded_object.scale.y, 0.4, rtol=1e-6)
    np.testing.assert_allclose(loaded_object.scale.z, 0.6, rtol=1e-6)

    loaded_object_location = loaded_object.location
    np.testing.assert_allclose(loaded_object_location.x, 4)
    np.testing.assert_allclose(loaded_object_location.y, 5)
    np.testing.assert_allclose(loaded_object_location.z, 6)


def test_file_saving_using_bsr_save(tmp_path):


    blend_file_path = tmp_path / "test_path"
    save(blend_file_path)  # Save using pathlib.Path object
    assert blend_file_path.exists()

    blend_file_path_str = (tmp_path / "test_str").as_posix()
    save(blend_file_path_str)  # Save using str object
    blend_file_path = Path(blend_file_path_str)
    assert blend_file_path.exists()


def test_file_reload_using_bsr_reload(blend_file):

    # Save the blend file into another path
    saved_blend_file = blend_file.parent / "test_path"
    save(saved_blend_file)

    # Change the radius and location of the object
    new_radius = 0.2
    obj = bpy.context.active_object
    scale_factor = 2
    # This scales the radius by 2; becomes 0.2
    obj.scale = (
        scale_factor * obj.scale[0],
        scale_factor * obj.scale[1],
        scale_factor * obj.scale[2],
    )
    obj.location = (1, 1, 1)

    # Reload/Revert the saved file
    reload(saved_blend_file)

    # read the object data
    # Radius and location of the object should be the same as the original file
    obj = bpy.context.active_object
    np.testing.assert_allclose(obj.dimensions[0] / 2, 0.1, rtol=1e-6)
    assert obj.location.x == 1
    assert obj.location.y == 2
    assert obj.location.z == 3


def test_file_not_found_reload():

    with pytest.raises(FileNotFoundError):
        reload("non_existent_file.blend")


@pytest.mark.parametrize(
    "name", [1, 1.0, (1, 2, 3), [1, 2, 3], {"a": 1}]
)  # Invalid types
def test_file_non_valid_path_type_for_reload(name):

    with pytest.raises(TypeError) as exec_info:
        reload(name)
    assert "should be either Path or str" in str(exec_info.value)


@pytest.mark.parametrize(
    "name", [1, 1.0, (1, 2, 3), [1, 2, 3], {"a": 1}]
)  # Invalid types
def test_file_save_non_path_object(name):

    with pytest.raises(TypeError) as exec_info:
        save(name)
    assert "should be either Path or str" in str(exec_info.value)


@pytest.mark.parametrize("name", ["test_str", pathlib.Path("test_path")])
def test_file_save_valid_path_pathlib(tmp_path, name):

    blend_file = tmp_path / name
    save(blend_file)
    assert blend_file.exists()


@pytest.mark.parametrize("name", ["test_str", pathlib.Path("test_path")])
def test_file_save_valid_path_str(tmp_path, name):

    blend_file = tmp_path / name
    save(blend_file.as_posix())
    assert blend_file.exists()
