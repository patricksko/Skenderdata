import blenderproc as bproc
import argparse
import os
import numpy as np
from pathlib import Path
import json
import random
import debugpy
import cv2

# debugpy.listen(5678)
# debugpy.wait_for_client()

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str, help='Path to the JSON configuration file for the dataset generation script')
args = parser.parse_args()
config_path = Path(args.config)

assert config_path.exists(), f'Config file {config_path} does not exist'
with open(config_path, 'r') as json_config_file:
    json_config = json.load(json_config_file)

config_dir = config_path
bop_dataset_path = config_dir / json_config["bop_path"]
cc_textures_path = config_dir / json_config["cc_textures_path"]
output_dir = config_dir / json_config["save_path"]

# Convert to absolute paths
bop_dataset_path = str(bop_dataset_path.resolve())
cc_textures_path = str(cc_textures_path.resolve())
output_dir = str(output_dir.resolve())

num_scenes = json_config['dataset']['num_scenes']
num_cameras = json_config['dataset']['num_cameras']

num_distractors = json_config['dataset']['num_distractors']
num_targets = json_config['dataset']['num_targets']
bproc.init()

# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'Legoblock'), model_type = 'cad', mm2m = True)

# load distractor bop objects
itodd_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'tyol'), mm2m = True)
ycbv_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'ycbv'), mm2m = True)
hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'hb'), mm2m = True)

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(bop_dataset_path, 'Legoblock'))

# set shading and hide objects
for obj in (target_bop_objs + itodd_dist_bop_objs + ycbv_dist_bop_objs + hb_dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)
    
# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(100)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)

# Define a function that samples 6-DoF poses
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
#bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])

for i in range(num_scenes):
    colors = [[*np.random.rand(3), 1.0] for _ in range(6)]

    # Sample bop objects for a scene
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=num_targets, replace=False))
    sampled_distractor_bop_objs = list(np.random.choice(itodd_dist_bop_objs, size=num_distractors, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(ycbv_dist_bop_objs, size=num_distractors, replace=False))
    sampled_distractor_bop_objs += list(np.random.choice(hb_dist_bop_objs, size=num_distractors, replace=False))

    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):     
        if obj in sampled_target_bop_objs:
            mat = bproc.material.create(f"target_material")
            obj.replace_materials(mat)
            mat.set_principled_shader_value("Base Color", random.choice(colors))
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.5, 0.9))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.1, 0.6))
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.hide(False)   
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)   
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))
        if obj.get_cp("bop_dataset_name") == 'itodd':  
            mat.set_principled_shader_value("Metallic", np.random.uniform(0.5, 1.0))
        if obj.get_cp("bop_dataset_name") == 'tless':
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.3, 1.0))
            mat.set_principled_shader_value("Metallic", np.random.uniform(0, 0.5))
        obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
        obj.hide(False)
    
    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3,6), 
                                    emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))  
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5,0.5,0.5],[1,1,1]))
    location = bproc.sampler.shell(center = [0, 0, 0], radius_min = 1, radius_max = 1.5,
                            elevation_min = 5, elevation_max = 89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)


    # Sample object poses and check collisions 
    bproc.object.sample_poses(objects_to_sample = sampled_target_bop_objs + sampled_distractor_bop_objs,
                            sample_pose_func = sample_pose_func, 
                            max_tries = 1000)
            
    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                    max_simulation_time=10,
                                                    check_object_interval=1,
                                                    substeps_per_frame = 20,
                                                    solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < num_cameras:
        # Sample location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.65,
                                radius_max = 0.94,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=6, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(output_dir, 'bop_data'),
                           target_objects = sampled_target_bop_objs,
                           dataset = 'Legoblock',
                           depth_scale = 0.1,
                           depths = data["depth"],
                           colors = data["colors"], 
                           color_file_format = "JPEG",
                           ignore_dist_thres = 10,
                           frames_per_chunk=num_cameras)
    
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):      
        obj.disable_rigidbody()
        obj.hide(True)