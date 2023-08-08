import bpy
from blender_scene_utils import Scene

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

# Add smoke domain to the scene
scene.add_smoke_domain(cube)
smoke_domain = scene.objects[-1]
# Modify smoke domain material properties
scene.modify_smoke_domain_material(smoke_domain)
# Modify smoke domain and effector properties
scene.modify_smoke_domain_properties(smoke_domain, (14, 5, 8), "//random_smokeplume")
scene.modify_smoke_effector_properties(cube)
# Animate the smoke
scene.animate_smoke(cube)
# Modify render properties
scene.modify_render_engine()
# Free all fluid simulations
bpy.ops.fluid.free_all()
# Increment the smoke plume folder number
base_folder = r'D:\mploi\Documents\Albatros\albatros\smoke_generator_V1\blender_images'
existing_folders = [folder for folder in os.listdir(base_folder) if folder.startswith('smokeplume_')]
highest_number = max([int(folder.split('_')[1]) for folder in existing_folders]) if existing_folders else 0
next_number = highest_number + 1
next_folder = f'smokeplume_{next_number:02d}'
# Set the cache directory for the "Smoke Domain" object's "Fluid" modifier
bpy.data.objects['Smoke Domain'].modifiers['Fluid'].domain_settings.cache_directory = f"//cache_directories/{next_folder}/"
# Start the fluid simulation bake
bpy.ops.fluid.bake_all()
# Set the render output file path for the current scene
bpy.context.scene.render.filepath = f"//../smoke_images/{next_folder}/"
# Print the new folder path for reference
print(f"New smoke plume folder: {next_folder}")