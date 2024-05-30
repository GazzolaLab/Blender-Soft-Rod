#This part will work either minimized like this or with the entire blender script

import bpy

#Opens file; Preset with previously saved file's filepath
bpy.ops.wm.open_mainfile(filepath="/Users/rohitharish/Downloads/Blender Summer .blend files/rodsimvscode2.blend")   

#To open file in Blender, copy the line below into last line of terminal; Last part is the script file's name (Stored in my downloads section)

#/Applications/Blender.app/Contents/MacOS/Blender --python "/Users/rohitharish/Downloads/open_blend_file.py"
