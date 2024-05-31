import pathlib

import bpy
import pytest


# Test file creation
def test_file_create_using_bpy(tmp_path):
    blend_file = tmp_path / "test.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
    assert blend_file.exists()


@pytest.fixture(scope="session")
def blend_file(tmp_path_factory):
    blend_file = tmp_path_factory.mktemp("data") / "test.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_file))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(0, 0, 0))
    return blend_file


def test_file_opening_using_bpy(blend_file):
    # TODO: Open the blend file and load the object
    ...

    loaded_object_radius = None
    loaded_object_location = None
    assert loaded_object_radius == 0.1
    assert loaded_object_location == (0, 0, 0)


def test_file_opening_and_writing_data_using_bpy(blend_file):
    # TODO: Open the blend file
    ...

    # TODO: Change the radius of the object
    new_radius = 0.2
    ...

    # TODO: Change the location of the object
    new_location = (1, 1, 1)
    ...

    # TODO: Save the blend file in different name
    new_blend_file = blend_file.parent / "test2.blend"
    ...
    assert new_blend_file.exists()

    # TODO: Open the new blend file and load the object
    loaded_object_radius = None
    loaded_object_location = None
    assert loaded_object_radius == new_radius
    assert loaded_object_location == new_location


def test_file_saving_using_bsr_save(tmp_path):
    from bsr.file import save

    blend_file = tmp_path / "test.blend"
    save(blend_file)  # Save using pathlib.Path object
    assert blend_file.exists()

    blend_file = (tmp_path / "test2.blend").as_posix()
    save(blend_file)  # Save using str object
    assert blend_file.exists()


def test_file_reload_using_bsr_reload(blend_file):
    from bsr.file import reload, save

    # Save the blend file into another path
    saved_blend_file = blend_file.parent / "test2.blend"
    save(saved_blend_file)

    # Change the radius and location of the object
    new_radius = 0.2
    new_location = (1, 1, 1)
    obj = bpy.context.active_object
    obj_radius = new_radius
    obj.location = new_location

    # Reload the saved file
    reload(saved_blend_file)

    # read the object data
    # Radius and location of the object should be the same as the original file
    obj = bpy.context.active_object
    assert obj.radius == 0.1
    assert obj.location == (0, 0, 0)


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
def test_file_save_non_path_object(name: str | pathlib.Path):
    from bsr.file import save

    with pytest.raises(ValueError):
        save(name)


@pytest.mark.parametrize("name", ["test.blend", pathlib.Path("test.blend2")])
def test_file_save_valid_path_pathlib(tmp_path, name):
    from bsr.file import save

    blend_file = tmp_path / name
    save(blend_file)
    assert blend_file.exists()


@pytest.mark.parametrize("name", ["test.blend", pathlib.Path("test.blend2")])
def test_file_save_valid_path_str(tmp_path, name):
    from bsr.file import save

    blend_file = tmp_path / name
    save(blend_file.as_posix())
    assert blend_file.exists()
