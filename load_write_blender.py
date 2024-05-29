import os

import bpy
import bsr


def main():

    load_filepath = "Blender/3rodbr2.blend"

    # load a single object
    with bpy.data.libraries.load(load_filepath, relative=True) as (data_from, data_to):
        data_to.scenes = ["Scene"]
    
    # load all meshes
    with bpy.data.libraries.load(load_filepath) as (data_from, data_to):
        data_to.meshes = data_from.meshes


    # link all objects starting with 'A'
    with bpy.data.libraries.load(load_filepath, link=True) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if name.startswith("A")]


    # append everything
    with bpy.data.libraries.load(load_filepath) as (data_from, data_to):
        for attr in dir(data_to):
            setattr(data_to, attr, getattr(data_from, attr))


    # the loaded objects can be accessed from 'data_to' outside of the context
    # since loading the data replaces the strings for the datablocks or None
    # if the datablock could not be loaded.
    with bpy.data.libraries.load(load_filepath) as (data_from, data_to):
        data_to.meshes = data_from.meshes
    # now operate directly on the loaded data
    for mesh in data_to.meshes:
        if mesh is not None:
            print(mesh.name)


    # TODO: The writing part is not working
    write_filepath = "/Blender/3rodbr2_write.blend"
    bpy.data.libraries.write(write_filepath, set(data_to), path_remap="RELATIVE")

    
    


if __name__ == "__main__":
    main()