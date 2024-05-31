import os

import bpy

import bsr


def main():

    load_filepath = "Blender/3rodbr2.blend"

    with bpy.data.libraries.load(load_filepath, link=True) as (
        data_from,
        data_to,
    ):
        data_to.objects = [
            name for name in data_from.objects if name.startswith("S")
        ]

    for obj in data_to.objects:
        assert obj is not None
        print(obj.name)

    # TODO: The writing part is not working
    # data_to.objects["Cube"].select_set(True)
    write_filepath = "/Blender/3rodbr2_write.blend"
    bpy.data.libraries.write(
        write_filepath, set(bpy.context.selected_objects), path_remap="RELATIVE"
    )


if __name__ == "__main__":
    main()
