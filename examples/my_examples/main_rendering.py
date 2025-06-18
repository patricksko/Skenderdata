import blenderproc as bproc
from blenderproc.python.loader import TextureLoader
import open3d as o3d
import argparse
import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
# import debugpy

# debugpy.listen(5678)
# debugpy.wait_for_client()

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str, help='Path to the JSON configuration file for the dataset generation script')
parser.add_argument('--draw_bbox', action='store_true', help="Draw 2D bounding boxes on images if set")
args = parser.parse_args()
config_path = Path(args.config)

assert config_path.exists(), f'Config file {config_path} does not exist'
with open(config_path, 'r') as json_config_file:
    json_config = json.load(json_config_file)

bop_dataset_path = os.path.normpath(os.path.join(config_path, json_config["bop_path"]))
cc_textures_path = os.path.normpath(os.path.join(config_path, json_config["cc_textures_path"]))

num_scenes = json_config['dataset']['num_scenes']
num_cameras = json_config['dataset']['num_cameras']

output_dir = os.path.normpath(os.path.join(config_path, json_config["save_path"]))


bproc.init()

# Load BOP objects
#target_bop_objs = bproc.loader.load_obj(filepath=os.path.join(args.bop_parent_path, 'Legoblock'), object_model_unit='mm')
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(bop_dataset_path, 'Legoblock'), object_model_unit='mm', obj_ids=[1, 2, 3])
tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(bop_dataset_path, 'tless'), model_type='cad', object_model_unit='mm')
hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(bop_dataset_path, 'hb'), object_model_unit='mm')
tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path=os.path.join(bop_dataset_path, 'tyol'), object_model_unit='mm')
print("Dataloading accomplished")

# Load intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(bop_dataset_path, 'ycbv'))

colors = [[*np.random.rand(3), 1.0] for _ in range(6)]



# Set shading and hide objects
for obj in (target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs):
    obj.set_shading_mode('auto')
    obj.hide(True)

# Create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

# Lights
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_point = bproc.types.Light()
lp_intensity = np.random.uniform(json_config['scene']['lights']['min_intensity'], json_config['scene']['lights']['max_intensity'])
light_point.set_energy(lp_intensity)

# Load textures
#cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)
cc_texture_path = os.path.join(cc_textures_path, "TU_floor/ALU_Desk.jpeg")
cc_textures = bproc.material.create_material_from_texture(cc_texture_path, "ALU_Desk")


# # Optionally set additional properties
# cc_texture.set_principled_shader_value("Roughness", 0.8)
# cc_texture.set_principled_shader_value("Specular", 0.2)
print("Loaded Texture")
def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)

for i in range(num_scenes):
    num_obj = np.random.randint(json_config['scene']['objects']['min_count'], json_config['scene']['objects']['max_count'])
    print(num_obj)
    sampled_target_bop_objs = list(np.random.choice(target_bop_objs, size=num_obj, replace=False))
    num_obj = np.random.randint(json_config['scene']['objects']['min_count'], json_config['scene']['objects']['max_count'])
    sampled_distractor_bop_objs = list(np.random.choice(tless_dist_bop_objs, size=num_obj, replace=False))
    num_obj = np.random.randint(json_config['scene']['objects']['min_count'], json_config['scene']['objects']['max_count'])
    sampled_distractor_bop_objs += list(np.random.choice(hb_dist_bop_objs, size=num_obj, replace=False))
    num_obj = np.random.randint(json_config['scene']['objects']['min_count'], json_config['scene']['objects']['max_count'])
    sampled_distractor_bop_objs += list(np.random.choice(tyol_dist_bop_objs, size=num_obj, replace=False))

    for obj, color in zip(sampled_target_bop_objs, colors):
        obj.set_shading_mode('auto')

        # Clone the material to avoid shared material issues
        mats = obj.get_materials()
        if mats:
            original_mat = mats[0]
            new_mat = original_mat
            obj.replace_materials(new_mat)

            new_mat.set_principled_shader_value("Base Color", color)
            new_mat.set_principled_shader_value("Roughness", 0.8)
            new_mat.set_principled_shader_value("Specular IOR Level", 0.1)

        obj.hide(False)
        
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
        mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
        obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.hide(False)
        

    light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                       emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5, elevation_min=5, elevation_max=89)
    light_point.set_location(location)

    random_cc_texture = cc_textures#np.random.choice(cc_textures)
    for plane in room_planes:
        print("Texture for room")
        plane.replace_materials(random_cc_texture)

    bproc.object.sample_poses(objects_to_sample=sampled_target_bop_objs + sampled_distractor_bop_objs,
                              sample_pose_func=sample_pose_func, max_tries=1000)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                      max_simulation_time=10,
                                                      check_object_interval=1,
                                                      substeps_per_frame=20,
                                                      solver_iters=25)

    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(sampled_target_bop_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < num_cameras:
        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.61, radius_max=1.24, elevation_min=5, elevation_max=89)
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=14, replace=True))
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi, np.pi))
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # RENDER SCENE
    data = bproc.renderer.render()

    # =================== Bounding Box Overlay Code ===================
    if json_config["draw_bbox"]:
        bbox_overlay_dir = os.path.join(output_dir, "bbox_overlay")
        os.makedirs(bbox_overlay_dir, exist_ok=True)

        for frame_idx, rgb_img in enumerate(data["colors"]):
            img = np.array(rgb_img)

            for obj in sampled_target_bop_objs:
                points = bproc.camera.project_points(obj.get_bound_box(), frame=frame_idx)

                for p in points:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                        img[y-2:y+3, x-2:x+3] = [255, 0, 0]

                min_xy = np.min(points, axis=0).astype(int) #top left corner
                max_xy = np.max(points, axis=0).astype(int) #bottom right corner
                if (0 <= min_xy[0] < img.shape[1] and 0 <= max_xy[0] < img.shape[1] and
                    0 <= min_xy[1] < img.shape[0] and 0 <= max_xy[1] < img.shape[0]):
                    img[min_xy[1]:max_xy[1], min_xy[0]] = [0, 255, 0]
                    img[min_xy[1]:max_xy[1], max_xy[0]] = [0, 255, 0]
                    img[min_xy[1], min_xy[0]:max_xy[0]] = [0, 255, 0]
                    img[max_xy[1], min_xy[0]:max_xy[0]] = [0, 255, 0]

            Image.fromarray(img).save(os.path.join(bbox_overlay_dir, f"{i:04d}_{frame_idx:02d}.jpg"))
    # ================================================================
    depth_scale = 0.1  # 1 mm per unit
    scaled_depths = [(depth / depth_scale).astype(np.uint16) for depth in data["depth"]]

    # Overwrite depth in the data dict
    data["depth"] = scaled_depths

    scene_dir = os.path.join(output_dir, f"scene_{i}")
    os.makedirs(scene_dir, exist_ok=True)

    bproc.writer.write_hdf5(scene_dir, data)
    # Write data in bop format
    # bproc.writer.write_bop(os.path.join(output_dir, 'bop_data'),
    #                        target_objects=sampled_target_bop_objs,
    #                        dataset='ycbv',
    #                        depth_scale=0.1,
    #                        depths=data["depth"],
    #                        colors=data["colors"],
    #                        color_file_format="JPEG",
    #                        ignore_dist_thres=10)

    # Hide and clean up
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs):
        obj.disable_rigidbody()
        obj.hide(True)
        