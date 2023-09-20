import bpy
import sys
import os

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )
    #print(sys.path)
from blender_scene_utils import Scene

##-----------------------------------------------------------------------------------------
##                        Blender Main script for Smoke Simulation
##-----------------------------------------------------------------------------------------
# Modify to your output path, should be ABSOLUTE
OUPUT_PATH = r"D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\blender_images"
# Create the scene instance
bpy.ops.file.make_paths_relative()
# Create the scene instance
scene = Scene()
# Delete all existing objects and animations
scene.delete_all_objects()
scene.clear_all_actions()
# Create the cube
scene.create_cube((0, 0, 0))
cube = scene.objects[-1]
# Create the wind
scene.create_wind()
wind = scene.objects[-1]
# Create the lights
scene.create_light((0, 3.7, 10), (-23.3, 0, 0))
light_1 = scene.objects[-1]
scene.create_light((-20, 2.4, 4), (142, -117, -140))
light_2 = scene.objects[-1]
scene.create_light((20, 3, 5), (150, -240, -208))
light_3 = scene.objects[-1]
# Create the camera
scene.create_camera((0, -40, 29), (58.7, 0, 0), 70)
camera = scene.objects[-1]
bpy.context.scene.camera = camera

# Increment the smoke plume folder number
if not(os.path.exists(OUPUT_PATH)):
    os.mkdir(OUPUT_PATH)
    print(f"Output directory {OUPUT_PATH} Created")
existing_folders = [folder for folder in os.listdir(OUPUT_PATH) if folder.startswith('smokeplume_')]
highest_number = max([int(folder.split('_')[1]) for folder in existing_folders]) if existing_folders else 0
next_number = highest_number + 1
next_folder = f'smokeplume_{next_number:02d}'
# Set the render output file path for the current scene
bpy.context.scene.render.filepath = f"//../blender_images/{next_folder}/"
# Print the new folder path for reference
print(f"New smoke plume folder: {next_folder}")
# Add smoke domain to the scene
scene.add_smoke_domain(cube)
smoke_domain = scene.objects[-1]
# Modify smoke domain material properties
scene.modify_smoke_domain_material(smoke_domain)
# Modify smoke domain and effector properties
scene.modify_smoke_domain_properties(smoke_domain, (14, 5, 8), f"//random_smokeplume/{next_folder}")
scene.modify_smoke_effector_properties(cube)
# Animate the smoke
scene.animate_smoke(cube)
# Modify render properties
scene.modify_render_engine()

# Start the fluid simulation bake
#bpy.ops.fluid.bake_all()

# Render the animation
#bpy.ops.render.render(animation=True)
