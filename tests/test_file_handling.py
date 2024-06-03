import pathlib

import bpy
import numpy as np
import pytest

# Had to do an extra import (pip install mathutils in Terminal); Reason shown later
from mathutils import Vector


# Test file creation
def test_file_create_using_bpy(tmp_path):
    blend_file = tmp_path / "test.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
    assert blend_file.exists()


@pytest.fixture(scope="session")
def blend_file(tmp_path_factory):
    blend_file_path = tmp_path_factory.mktemp("data") / "test.blend"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0))
    # Moved save_as_mainfile line below creation of sphere; Previously rendered empty .blend file
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file_path))
    return blend_file_path


def test_file_opening_using_bpy(blend_file):
    # TODO: Open the blend file and load the object
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))
    objects = bpy.context.scene.objects
    loaded_object = objects.get("Sphere")
    # You cannot directly access the sphere's radius
    # Must estimate through dimensions of object's bounding box
    loaded_object_dimensions = loaded_object.dimensions
    # Dimension of the box on any axis is the approx sphere diameter, can divide by 2 for approx radius
    # Seems that the approximation is close but not quite correct
    # 0.09999993443489075 vs 0.1
    loaded_object_radius = loaded_object_dimensions[0] / 2
    # Location is displayed in the form Vector((0,0,0)), which is in Mathutils package
    loaded_object_location = loaded_object.location
    exact_radius = 0.1
    # Assert statement adjusted
    np.testing.assert_allclose(loaded_object_radius, exact_radius, atol=0.01)
    # Assert statement changed from Tuple to Vector to match Blender format
    assert loaded_object_location == Vector((0, 0, 0))


def test_file_opening_and_writing_data_using_bpy(blend_file):
    # TODO: Open the blend file
    bpy.ops.wm.open_mainfile(filepath=str(blend_file))

    # TODO: Change the radius of the object
    new_radius = 0.2
    # - Note; cannot driectly change sphere radius post-definition; can work around by scaling it
    objects = bpy.context.scene.objects
    loaded_object = objects.get("Sphere")
    scale_factor = 2
    loaded_object.scale = (
        scale_factor * loaded_object.scale[0],
        scale_factor * loaded_object.scale[1],
        scale_factor * loaded_object.scale[2],
    )
    # This effectively scales in each axis by the scale factor (2), increasing radius to 0.2

    # TODO: Change the location of the object
    new_location = Vector((1, 1, 1))
    loaded_object.location = new_location

    # TODO: Save the blend file in different name
    new_blend_file = blend_file.parent / "testname"
    bpy.ops.wm.save_as_mainfile(filepath=str(new_blend_file))
    assert new_blend_file.exists()

    # TODO: Open the new blend file and load the object
    bpy.ops.wm.save_as_mainfile(filepath=str(new_blend_file))
    loaded_object_dimensions = loaded_object.dimensions
    loaded_object_radius = loaded_object_dimensions[0] / 2
    loaded_object_location = loaded_object.location
    np.testing.assert_allclose(loaded_object_radius, new_radius, atol=0.01)
    assert loaded_object_location == new_location


def test_file_saving_using_bsr_save(tmp_path):
    from pathlib import Path

    from bsr.file import save

    blend_file_path = tmp_path / "test_path"
    save(blend_file_path)  # Save using pathlib.Path object
    assert blend_file_path.exists()

    blend_file_path_str = (tmp_path / "test_str").as_posix()
    save(blend_file_path_str)  # Save using str object
    blend_file_path = Path(blend_file_path_str)
    assert blend_file_path.exists()


def test_file_reload_using_bsr_reload(blend_file):
    from bsr.file import reload, save

    # Save the blend file into another path
    saved_blend_file = blend_file.parent / "test_path"
    save(saved_blend_file)

    # Change the radius and location of the object
    new_radius = 0.2
    new_location = Vector((1, 1, 1))
    obj = bpy.context.active_object
    scale_factor = 2
    obj.scale = (
        scale_factor * obj.scale[0],
        scale_factor * obj.scale[1],
        scale_factor * obj.scale[2],
    )
    object_radius = obj.dimensions[0] / 2
    obj.location = new_location

    # Reload the saved file
    reload(saved_blend_file)

    # read the object data
    # Radius and location of the object should be the same as the original file
    obj = bpy.context.active_object
    np.testing.assert_allclose(obj.dimensions[0] / 2, 0.1, atol=0.01)
    assert obj.location == Vector((0, 0, 0))


def test_file_not_found_reload():
    from bsr.file import reload

    with pytest.raises(FileNotFoundError):
        reload("non_existent_file.blend")


@pytest.mark.parametrize(
    "name", [1, 1.0, (1, 2, 3), [1, 2, 3], {"a": 1}]
)  # Invalid types
def test_file_non_valid_path_type_for_reload(name):
    from bsr.file import reload

    with pytest.raises(ValueError):
        reload(name)


@pytest.mark.parametrize(
    "name", [1, 1.0, (1, 2, 3), [1, 2, 3], {"a": 1}]
)  # Invalid types
def test_file_save_non_path_object(name):
    from bsr.file import save

    with pytest.raises(ValueError):
        save(name)


@pytest.mark.parametrize("name", ["test_str", pathlib.Path("test_path")])
def test_file_save_valid_path_pathlib(tmp_path, name):
    from bsr.file import save

    blend_file = tmp_path / name
    save(blend_file)
    assert blend_file.exists()


@pytest.mark.parametrize("name", ["test_str", pathlib.Path("test_path")])
def test_file_save_valid_path_str(tmp_path, name):
    from bsr.file import save

    blend_file = tmp_path / name
    save(blend_file.as_posix())
    assert blend_file.exists()
