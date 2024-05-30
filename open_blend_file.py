#This part will work either minimized like this or with the entire blender script
import os
import bpy

#Opens file; Preset with previously saved file's filepath

load_filepath = "3rodbr2_write.blend"
load_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), load_filepath)
bpy.ops.wm.open_mainfile(filepath=load_filepath)   
objects = bpy.context.scene.objects
for obj in objects:
    print(obj.name)

#To open file in Blender, copy the line below into last line of terminal; Last part is the script file's name (Stored in my downloads section)

#/Applications/Blender.app/Contents/MacOS/Blender --python "/Users/rohitharish/Downloads/open_blend_file.py"
