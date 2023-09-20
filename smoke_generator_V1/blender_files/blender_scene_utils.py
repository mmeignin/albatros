import bpy
import random 
import os 
import shutil
import math

##-----------------------------------------------------------------------------------------
##                        Blender Scene class for Smoke Simulation
##-----------------------------------------------------------------------------------------

class Scene:
    def __init__(self):
        self.objects = []
        
    def create_cube(self, location):
        """
        Create a small, smooth cube effector for the smoke simulation.
        This cube will control the smoke behavior and appearance.
        """
        bpy.ops.mesh.primitive_cube_add(location=location)
        cube = bpy.context.object
        cube.scale = (0.5, 0.5, 0.5) 
        subsurf_modifier = cube.modifiers.new(name="Subdivision", type='SUBSURF')
        subsurf_modifier.levels = 2

        self.objects.append(cube)

    def create_light(self,location,rotation):
        """
        Create an area light with default settings and add it to the scene.
        This method creates an area light that illuminates the scene.
        """
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object
        light.rotation_euler = [ math.radians(angle) for angle in rotation ]
        light.data.energy = 1000
        light.data.shape = 'RECTANGLE'
        light.data.size = 1
        light.data.size_y = 1 
        self.objects.append(light)

    def create_camera(self, location,rotation,scale):
        """
        Create a camera at the specified location with default orientation 
        """
        bpy.ops.object.camera_add(enter_editmode=False, align='WORLD')
        camera = bpy.context.object
        camera.location = location
        camera.rotation_euler = [math.radians(angle) for angle in rotation]
        camera.data.type ='PERSP'
        camera.data.ortho_scale = scale
        self.objects.append(camera)
    
    def create_wind(self):
        """
        Method generates a wind force to create more dynamic and varied smoke behavior.)
        The wind force is created with random properties such as location, strength, and noise to introduce randomness in the smoke simulation and make it more visually interesting.
        """
        # Define the minimum and maximum values for the wind's location in the 3D space
        min_x, max_x = -5, 5
        min_y, max_y = -5, 5
        min_z, max_z = 0, 10
        # Generate a random location for the wind within the defined range
        location = (
            random.uniform(min_x, max_x),
            -5,
            random.uniform(min_z, max_z)
        )
        # Define the minimum and maximum values for the wind's strength
        min_strength, max_strength = 4,12 
        # Create the wind effector at the randomly generated location and set its scale to (1, 1, 1)
        bpy.ops.object.effector_add(type='WIND', enter_editmode=False, align='WORLD', location=location, scale=(1, 1, 1))
        wind = bpy.context.object
        # Generate random rotation angles for the wind's orientation
        rotation = (
            random.uniform(180,360) ,
            random.uniform(-90, 90),
           0
        )
        # Set the wind's rotation using the generated rotation angles
        wind.rotation_euler = [math.radians(angle) for angle in rotation]
        # Set the wind's strength to a random value within the defined range
        wind.field.strength = random.uniform(min_strength, max_strength)
        # Set the wind's noise parameter to introduce variation in the wind force
        wind.field.noise = 1
        # Add the wind effector object to the list of objects in the scene
        self.objects.append(wind)

    def add_smoke_domain(self, smoke_effector):
        """
        This method creates a smoke domain and smoke effector for the given object. The smoke domain
        delimits the smoke's properties and behavior, while the smoke effector emits the smoke particles.
        """
        bpy.ops.object.select_all(action='DESELECT')
        smoke_effector.select_set(True)
        bpy.context.view_layer.objects.active = smoke_effector
        bpy.ops.object.quick_smoke()
        smoke_domain = bpy.context.object
        self.objects.append(smoke_domain)

    def clear_cache(self,cache_directory):
        """
        Clear the cache by deleting cache files and directories.
        """
        blend_file_directory = os.path.dirname(bpy.data.filepath)
        cache_directory_norm = cache_directory.replace("//", "")
        absolute_cache_directory = os.path.join(blend_file_directory,cache_directory_norm).replace("\\","/")
        print(absolute_cache_directory,os.path.exists(absolute_cache_directory))
        if os.path.exists(absolute_cache_directory):
            try:
                shutil.rmtree(absolute_cache_directory)
                print(f"Cache directory '{absolute_cache_directory}' deleted.")
            except Exception as e:
                print(f"Failed to delete cache directory: {e}")

    def modify_smoke_domain_properties(self, smoke_domain, scale, cache_directory):
        """
        Modify smoke properties for better-looking smoke. 
        Important parameters are:
        - resolution_max: Maximum resolution of the smoke domain.
        - vorticity: Controls the swirling behavior of the smoke.
        - additional_res: Additional resolution added to the smoke domain.
        - use_noise: Enables noise in the smoke simulation.
        - use_collision_border_bottom: Emulate the ground.
        """
        smoke_domain_offset = (0, 0, 5)
        smoke_domain.location = smoke_domain_offset 

        # Check if the added object is the smoke domain
        if smoke_domain.type == 'MESH':
            smoke_domain_scale = scale
            # Apply the scale to the smoke domain object
            smoke_domain.scale = smoke_domain_scale
            # Access the fluid modifier
            smoke_domain_fluid = smoke_domain.modifiers.get("Fluid")
            if smoke_domain_fluid is not None:
                smoke_domain_fluid.domain_settings.resolution_max = 128
                smoke_domain_fluid.domain_settings.use_adaptive_domain = True
                smoke_domain_fluid.domain_settings.vorticity = 0.2
                smoke_domain_fluid.domain_settings.additional_res = 2 
                smoke_domain_fluid.domain_settings.use_noise = True
                smoke_domain_fluid.domain_settings.noise_scale = 2
                smoke_domain_fluid.domain_settings.noise_strength = 0.4
                smoke_domain_fluid.domain_settings.noise_pos_scale = 8
                smoke_domain_fluid.domain_settings.noise_time_anim = 1
                smoke_domain_fluid.domain_settings.openvdb_cache_compress_type = 'ZIP'
                smoke_domain_fluid.domain_settings.openvdb_data_depth = '32'
                smoke_domain_fluid.domain_settings.cache_frame_end = 50
                smoke_domain_fluid.domain_settings.cache_type = "ALL"
                smoke_domain_fluid.domain_settings.use_collision_border_bottom = True
                # Change cache directory
                smoke_domain_fluid.domain_settings.cache_directory = cache_directory
                self.clear_cache(cache_directory)
                smoke_domain_fluid.domain_settings.cache_directory = cache_directory
        else:
            print("The provided object is not a smoke domain.")
    
    def modify_smoke_effector_properties(self, smoke_effector):
        """
        Modify smoke properties for a better-looking smoke emission from the smoke effector.
        Important properties:
        - smoke_color: The color of the emitted smoke.
        - subframes: Number of subframes for smoother smoke animation.
        - surface_distance: Controls the emission surface distance for the smoke.
        - velocity_coord: Random initial velocity coordinates for the smoke.
        - density: Density of the smoke over time.
        The density of the smoke is animated over time to create a more realistic smoke behavior.
        """
        min_speed = 1
        max_speed = 50  
        # Check if the added object is the smoke domain
        if smoke_effector.type == 'MESH':
            # Access the fluid modifier
            smoke_effector_fluid = smoke_effector.modifiers.get("Fluid")
            if smoke_effector_fluid is not None:
                smoke_effector_fluid.flow_settings.smoke_color = (1, 1, 1)
                smoke_effector_fluid.flow_settings.subframes = 2
                smoke_effector_fluid.flow_settings.surface_distance = random.uniform(0.75,1.5)
                smoke_effector_fluid.flow_settings.volume_density = 1
                smoke_effector_fluid.flow_settings.use_initial_velocity = True
                smoke_effector_fluid.flow_settings.velocity_factor = 0
                smoke_effector_fluid.flow_settings.velocity_coord[0] = random.choice([-1, 1]) * random.uniform(min_speed, max_speed)
                smoke_effector_fluid.flow_settings.velocity_coord[1] = random.uniform(min_speed,max_speed/10)
                smoke_effector_fluid.flow_settings.velocity_coord[2] = random.choice([-1, 1]) * random.uniform(min_speed, max_speed)
                print(f"Initial Velocity {smoke_effector_fluid.flow_settings.velocity_coord[0],smoke_effector_fluid.flow_settings.velocity_coord[1],smoke_effector_fluid.flow_settings.velocity_coord[2]}")
                # Animate Density over time
                smoke_effector_fluid.flow_settings.density = 0.0
                smoke_effector_fluid.flow_settings.keyframe_insert(data_path="density", frame=0)
                smoke_effector_fluid.flow_settings.keyframe_insert(data_path="density", frame=14)
                smoke_effector_fluid.flow_settings.density = 1
                smoke_effector_fluid.flow_settings.keyframe_insert(data_path="density", frame=15)
                smoke_effector_fluid.flow_settings.keyframe_insert(data_path="density", frame=40)
        else:
            print("The provided object is not a smoke object.")

    def animate_smoke(self, smoke_effector):
        """
        The method adds noise to the smoke density over time, resulting in a more dynamic and varied smoke simulation
        """
        if smoke_effector.animation_data is None:
            print("The object has no animation data.")
            smoke_effector.animation_data_create()
        # Access the existing animation data
        anim_data = smoke_effector.animation_data
        active_action = anim_data.action
        fcurve = active_action.fcurves[0]
        # Add a new modifier to the F-Curve
        modifier = fcurve.modifiers.new(type='NOISE')
        # Set the properties of the 'NOISE' modifier
        modifier.strength = random.uniform(2, 5)
        modifier.scale = 1.0
        modifier.offset = 0.0
        modifier.blend_type = 'MULTIPLY'

    def modify_smoke_domain_material(self, smoke_domain):
        """
        Modify the appearance of smoke density during rendering using node-based materials.
        This method adjusts how smoke looks when rendered, making it denser or more transparent
        """
        # Check if the provided object is a mesh with a node-based material
        if smoke_domain.type != 'MESH' or not smoke_domain.active_material or not smoke_domain.active_material.use_nodes:
            print("The provided object is not a smoke domain with a node-based material.")
            return
        # Access the material and its nodes
        material = smoke_domain.active_material
        nodes = material.node_tree.nodes
        # Find the node responsible for controlling smoke density (principled volume node)
        principled_volume = nodes[1]
        # Change the material color of the smoke
        color_intensity = 1
        principled_volume.inputs[0].default_value = (color_intensity, color_intensity, color_intensity, 1)

        # Create attribute and math nodes to control density
        attribute_node = nodes.new(type='ShaderNodeAttribute')
        attribute_node.attribute_name = "Density"

        math_node_1 = nodes.new(type='ShaderNodeMath')
        math_node_1.operation = 'ADD' 
        math_node_1.inputs[1].default_value = 1
        
        math_node_2 = nodes.new(type='ShaderNodeMath')
        math_node_2.operation = 'MULTIPLY'  
        math_node_2.inputs[1].default_value = 1
        # Connect nodes to control density
        material.node_tree.links.new(attribute_node.outputs['Fac'], math_node_1.inputs[0])
        material.node_tree.links.new(math_node_1.outputs['Value'], math_node_2.inputs['Value'])
        material.node_tree.links.new(math_node_2.outputs['Value'], principled_volume.inputs['Density'])


    def modify_render_engine(self):
        """
        Modify Blender render properties to achieve a more accurate-looking smoke simulation.
        This method adjusts the settings used for rendering the smoke simulation to improve its appearance.
        """
        # Set the rendering engine to CYCLES for more realistic rendering
        bpy.context.scene.render.engine ='CYCLES'
        # Enable render border and cropping for a more focused render
        bpy.context.scene.render.use_border = True
        bpy.context.scene.render.use_crop_to_border = True
        # Enable film transparency to render smoke with a transparent background
        bpy.context.scene.render.film_transparent = True
        # Enable preview denoising to reduce noise during preview rendering
        bpy.context.scene.cycles.use_denoising = False
        # Adjust volume bounces to control the amount of light scattering in the smoke
        bpy.context.scene.cycles.volume_bounces = 2
        # Set the number of samples for a higher quality render
        bpy.context.scene.cycles.samples = 1024
        # Set the minimum number of adaptive samples to improve sampling efficiency
        bpy.context.scene.cycles.adaptive_min_samples = 128
        # Set the tile size for rendering efficiency
        bpy.context.scene.cycles.tile_size = 1024
        # Enable compact BVH for faster rendering
        bpy.context.scene.cycles.debug_use_compact_bvh = True
        # Hide the first object (smoke effector so the cube containing the smoke isn't rendered) from rendering
        self.objects[0].hide_render = True
        # Set the starting and ending frame for the smoke animation
        bpy.context.scene.frame_start = 14
        bpy.context.scene.frame_end = 50
        
    def delete_all_objects(self):
        """
        Delete all objects from the Blender scene.

        This method removes all objects present in the scene, providing a clean slate
        for creating new objects or running simulations
        """
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def clear_all_actions(self):
        """
        Remove all animation actions from the current Blender scene.
        """
        for action in bpy.data.actions:
            bpy.data.actions.remove(action)


 