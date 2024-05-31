# This part will work either minimized like this or with the entire blender script
import os

import bpy

# Opens file; Preset with previously saved file's filepath

load_filepath = "3rodbr2_write.blend"
load_filepath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), load_filepath
)
bpy.ops.wm.open_mainfile(filepath=load_filepath)
objects = bpy.context.scene.objects
for obj in objects:
    print(obj.name)
print(objects[2].location[0])
objects[2].location[0] = 100

print(objects[2].location[0])

write_filepath = "3rodbr2_write2.blend"
write_filepath = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), write_filepath
)
bpy.ops.wm.save_as_mainfile(filepath=write_filepath)
# To open file in Blender, copy the line below into last line of terminal; Last part is the script file's name (Stored in my downloads section)

# /Applications/Blender.app/Contents/MacOS/Blender --python "/Users/rohitharish/Downloads/open_blend_file.py"
