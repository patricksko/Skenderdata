import blenderproc as bproc
from blenderproc.python.loader import TextureLoader
import argparse
import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import yaml
import random
import debugpy
import png
from scipy.spatial.transform import Rotation as R


# debugpy.listen(5678)
# debugpy.wait_for_client()


def save_depth(path: str, im: np.ndarray):
        """Saves a depth image (16-bit) to a PNG file.
        From the BOP toolkit (https://github.com/thodan/bop_toolkit).

        :param path: Path to the output depth image file.
        :param im: ndarray with the depth image to save.
        """
        if not path.endswith(".png"):
            raise ValueError('Only PNG format is currently supported.')

        im[im > 65535] = 65535
        im_uint16 = np.round(im).astype(np.uint16)

        # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
        w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
        with open(path, 'wb') as f:
            w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))
def sample_pose_func(obj: bproc.types.MeshObject):
    """Sample random pose for an object"""
    min_loc = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max_loc = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min_loc, max_loc))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())

def mask_to_polygons(mask):
    """Convert binary mask to YOLO polygon format"""
    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) >= 3:  # Minimum 3 points for a valid polygon
            polygon = contour.reshape(-1, 2).flatten().tolist()
            if len(polygon) >= 6:  # Minimum 3 points (6 coordinates)
                polygons.append(polygon)
    return polygons

def visualize_mask_and_polygons(rgb_img, mask, polygons, obj_name, save_path):
    """Create a visualization showing the original image, binary mask, and polygon overlay"""
    img_height, img_width = mask.shape
    
    # Create a 2x2 subplot visualization
    fig_height, fig_width = img_height * 2, img_width * 2
    visualization = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)
    
    # Top-left: Original RGB image
    visualization[:img_height, :img_width] = rgb_img
    
    # Top-right: Binary mask (white on black)
    mask_rgb = np.stack([mask * 255] * 3, axis=-1).astype(np.uint8)
    visualization[:img_height, img_width:] = mask_rgb
    
    # Bottom-left: RGB image with mask overlay (semi-transparent red)
    overlay_img = rgb_img.copy()
    red_overlay = np.zeros_like(rgb_img)
    red_overlay[:, :, 0] = mask * 255  # Red channel
    overlay_img = cv2.addWeighted(overlay_img, 0.7, red_overlay, 0.3, 0)
    visualization[img_height:, :img_width] = overlay_img
    
    # Bottom-right: RGB image with polygon contours
    polygon_img = rgb_img.copy()
    for poly in polygons:
        # Convert normalized coordinates back to pixel coordinates
        points = []
        for i in range(0, len(poly), 2):
            x = int(poly[i] * img_width)
            y = int(poly[i + 1] * img_height)
            points.append([x, y])
        
        if len(points) >= 3:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(polygon_img, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Also fill the polygon with semi-transparent green
            polygon_overlay = np.zeros_like(polygon_img)
            cv2.fillPoly(polygon_overlay, [points], (0, 255, 0))
            polygon_img = cv2.addWeighted(polygon_img, 0.8, polygon_overlay, 0.2, 0)
    
    visualization[img_height:, img_width:] = polygon_img
    
    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_color = (255, 255, 255)
    
    # Add labels with black background for better visibility
    labels = ["Original Image", "Binary Mask", "Mask Overlay", "Polygon Contours"]
    positions = [(10, 30), (img_width + 10, 30), (10, img_height + 30), (img_width + 10, img_height + 30)]
    
    for label, pos in zip(labels, positions):
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        # Draw black background
        cv2.rectangle(visualization, (pos[0] - 5, pos[1] - text_height - 5), 
                     (pos[0] + text_width + 5, pos[1] + baseline + 5), (0, 0, 0), -1)
        # Draw text
        cv2.putText(visualization, label, pos, font, font_scale, text_color, font_thickness)
    
    # Add object name at the top
    obj_label = f"Object: {obj_name}"
    (text_width, text_height), baseline = cv2.getTextSize(obj_label, font, font_scale + 0.2, font_thickness)
    cv2.rectangle(visualization, (fig_width // 2 - text_width // 2 - 10, 5), 
                 (fig_width // 2 + text_width // 2 + 10, text_height + baseline + 10), (0, 0, 0), -1)
    cv2.putText(visualization, obj_label, (fig_width // 2 - text_width // 2, text_height + 10), 
                font, font_scale + 0.2, (0, 255, 255), font_thickness)
    
    # Save visualization
    Image.fromarray(visualization).save(save_path)



parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True, type=str, help='Path to the JSON configuration file for the dataset generation script')
parser.add_argument('--val_split_ratio', type=float, default=0.2, help='Ratio of validation data (default: 0.2)')
args = parser.parse_args()
config_path = Path(args.config)

# Class mapping: Lego_Block will be class ID 0 for YOLO
CLASS_NAME_TO_ID = {
    "Lego_Block": 0
}

assert config_path.exists(), f'Config file {config_path} does not exist'
with open(config_path, 'r') as json_config_file:
    json_config = json.load(json_config_file)

# Fix path resolution - use config file's parent directory as base
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

# Directories for images
image_train_dir = os.path.join(output_dir, "images", "train")
image_val_dir = os.path.join(output_dir, "images", "val")
os.makedirs(image_train_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)

depth_train_dir = os.path.join(output_dir, "depth", "train")
depth_val_dir = os.path.join(output_dir, "depth", "val")
os.makedirs(depth_train_dir, exist_ok=True)
os.makedirs(depth_val_dir, exist_ok=True)

# Directories for YOLO segmentation labels
label_seg_train_dir = os.path.join(output_dir, "labels", "train")
label_seg_val_dir = os.path.join(output_dir, "labels", "val")
os.makedirs(label_seg_train_dir, exist_ok=True)
os.makedirs(label_seg_val_dir, exist_ok=True)

# Directory for 2D bounding box overlays (for visual debugging if draw_bbox is true)
bbox_overlay_dir = os.path.join(output_dir, "bbox_overlay")
if json_config.get("draw_bbox", False):
    os.makedirs(bbox_overlay_dir, exist_ok=True)

# Directory for mask visualizations
mask_viz_dir = os.path.join(output_dir, "mask_visualizations")
os.makedirs(mask_viz_dir, exist_ok=True)



bproc.init()

# load bop objects into the scene
target_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'Legoblock'), object_model_unit='mm', obj_ids=[1, 2, 3, 4, 5, 6])

# load distractor bop objects
tless_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'tless'), model_type = 'cad', object_model_unit='mm')
hb_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'hb'), object_model_unit='mm')
tyol_dist_bop_objs = bproc.loader.load_bop_objs(bop_dataset_path = os.path.join(bop_dataset_path, 'tyol'), object_model_unit='mm')

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(bop_dataset_path = os.path.join(bop_dataset_path, 'ycbv'))

# set shading and hide objects
for obj in (target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs):
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
light_point.set_energy(200)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
    
# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)
bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])

global_object_counter = 0

for i in range(num_scenes):
    print(f"Generating scene {i+1}/{num_scenes}")
    
    # Generate random colors
    colors = [[*np.random.rand(3), 1.0] for _ in range(6)]
    # Sample target objects with proper copying
    sampled_target_bop_objs = []
    for _ in range(5):
        original_obj = random.choice(target_bop_objs)
        # Create a copy of the object
        copied_obj = original_obj.duplicate()
        copied_obj.set_name(f"target_obj_{global_object_counter}")
        global_object_counter += 1
        sampled_target_bop_objs.append(copied_obj)

    # Sample distractor objects with proper copying
    sampled_distractor_bop_objs = []
    
    for dist_objs, num_dist, prefix in [
        (tless_dist_bop_objs, 0, "tless"),
        (hb_dist_bop_objs, 0, "hb"),
        (tyol_dist_bop_objs, 0, "tyol")
    ]:
        if dist_objs and num_dist > 0:
            for _ in range(num_dist):
                original_obj = random.choice(dist_objs)
                copied_obj = original_obj.duplicate()
                copied_obj.set_name(f"distractor_{prefix}_obj_{global_object_counter}")
                global_object_counter += 1
                sampled_distractor_bop_objs.append(copied_obj)
    # Randomize materials and set physics
    for obj in (sampled_target_bop_objs + sampled_distractor_bop_objs): 
        if obj in sampled_target_bop_objs:
            # Target objects: random texture or color
            # if random.random() < 0.2 and custom_texture_mats:
            #     random_texture = random.choice(custom_texture_mats)
            #     # Create a copy of the material to avoid conflicts
            #     mat_copy = random_texture.duplicate()
            #     obj.replace_materials(mat_copy)
            #     mat_copy.set_principled_shader_value("Roughness", np.random.uniform(0.2, 0.9))
            #     mat_copy.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.0, 0.5))
            #else:
                # Create new material for this object
            mat = bproc.material.create(f"target_mat_{global_object_counter}")
            obj.replace_materials(mat)
            mat.set_principled_shader_value("Base Color", random.choice(colors))
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.5, 0.9))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.1, 0.6))
            obj.enable_rigidbody(True, mass=1.0, friction = 100.0, linear_damping = 0.99, angular_damping = 0.99)
            obj.hide(False)
        else:
            mat = obj.get_materials()[0]
            if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
                grey_col = np.random.uniform(0.1, 0.9)   
                mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])        
            mat.set_principled_shader_value("Roughness", np.random.uniform(0, 1.0))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0, 1.0))
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
        # Sample camera location
        location = bproc.sampler.shell(center = [0, 0, 0],
                                radius_min = 0.61,
                                radius_max = 1.24,
                                elevation_min = 5,
                                elevation_max = 89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(sampled_target_bop_objs, size=4, replace=False))
        # Compute rotation based on vector going from camera towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world = bproc.math.build_transformation_mat(location, rotation_matrix)
        with open("H_C_W", "a") as f:
            f.write(f"Camera frame {cam_poses}: \n")
            f.write(f"Homogenous T: {cam2world} \n")
        

        
        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world, frame=cam_poses)
            cam_poses += 1

        objs = bproc.object.get_all_mesh_objects()
        target_objs = [obj for obj in objs if obj in sampled_target_bop_objs]
        # camera_pose = bproc.camera.get_camera_pose()
    
        # with open("object_poses_in_camera.txt", "a") as f:
        #     for obj in target_objs:
        #         obj2world = obj.get_local2world_mat()
        #         world2cam = np.linalg.inv(cam2world)  # World to camera
        #         obj2cam_matrix = world2cam @ obj2world  # Object to camera (correct!)
        #         translation = obj2cam_matrix[:3, 3]
        #         rotation = obj2cam_matrix[:3, :3]
        #         quat = R.from_matrix(rotation).as_quat()
                
        #         f.write(f"Camera frame {cam_poses}, Object {obj.get_name()}: \n")
        #         f.write(f"Translation: {translation.tolist()}\n")
        #         f.write(f"Quaternion: {quat.tolist()}\n")
        #         f.write("\n")
    # render the whole pipeline
    data = bproc.renderer.render()
    K = bproc.camera.get_intrinsics_as_K_matrix()
    # Get image dimensions
    img_height, img_width, _ = data["colors"][0].shape
    # Process each rendered frame
    for frame_idx in range(len(data["colors"])):
        rgb_img = data["colors"][frame_idx].copy()

        # Get camera pose for this frame
        H_C_W = bproc.camera.get_camera_pose(frame_idx)
        H_W_C = np.linalg.inv(H_C_W)
        
        
        # Project each target object to image coordinates
        for obj in target_objs:
            H_O_W = obj.get_local2world_mat()
            with open("H_O_W.txt", "a") as f:
                f.write(f"Object {obj.get_name()}: \n")
                f.write(f"THomogenous T {H_O_W}\n")
            # H_W_O = np.linalg.inv(H_O_W)
            # H_C_O = H_W_C @ H_O_W
            # H_O_C = np.linalg.inv(H_C_O)
            # obj_pos_cam = H_C_O[:3, 3]
            # if obj_pos_cam[2] < 0:
            #     # Project 3D point to 2D image coordinates
            #     # Convert to homogeneous coordinates
            #     obj_pos_homogeneous = np.array([obj_pos_cam[0], obj_pos_cam[1], obj_pos_cam[2]])
                
            #     # Project using camera intrinsics: pixel = K * [X/Z, Y/Z, 1]
            #     pixel_coords = K @ obj_pos_homogeneous
            #     pixel_coords = pixel_coords / (pixel_coords[2])  # Normalize by Z
                
            #     # Convert to integer pixel coordinates
            #     u = int(img_width/2 - (int(pixel_coords[0])-img_width/2)) # x coordinate
            #     v = int(pixel_coords[1])  # y coordinate
                
            #     # Check if point is within image bounds
            #     if 0 <= u < img_width and 0 <= v < img_height:
            #         # Draw red circle at object position
            #         cv2.circle(rgb_img, (u, v), radius=5, color=(0, 0, 255), thickness=10)  # BGR format
                    
            #         # Optionally add object name as text
            #         cv2.putText(rgb_img, obj.get_name(), (u + 10, v - 10), 
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Save or display the annotated image
        cv2.imwrite(f"annotated_frame_{frame_idx:04d}.png", rgb_img)
    exit()
#         depth_img = data["depth"][frame_idx]
#         depth_mm = 1000.0 * depth_img  # [m] -> [mm]

#         # Get segmentation data
#         seg_mask_instances = data["instance_segmaps"][frame_idx]
        
#         # Build instance name mapping
#         instance_name_map = {}
#         if "instance_attribute_maps" in data and len(data["instance_attribute_maps"]) > frame_idx:
#             for item in data["instance_attribute_maps"][frame_idx]:
#                 instance_name_map[item["idx"]] = item["name"]

#         image_name = f"scene{i:04d}_cam{frame_idx:02d}.jpg"

#         # Determine split (train/val)
#         if np.random.rand() < args.val_split_ratio:
#             split = 'val'
#             image_save_path = os.path.join(image_val_dir, image_name)
#             label_save_path = os.path.join(label_seg_val_dir, image_name.replace('.jpg', '.txt'))
#             depth_save_path = os.path.join(depth_val_dir, image_name.replace('.jpg', '.png'))
#         else:
#             split = 'train'
#             image_save_path = os.path.join(image_train_dir, image_name)
#             label_save_path = os.path.join(label_seg_train_dir, image_name.replace('.jpg', '.txt'))
#             depth_save_path = os.path.join(depth_train_dir, image_name.replace('.jpg', '.png'))

#         # Save image
#         Image.fromarray(rgb_img).save(image_save_path)
#         save_depth(depth_save_path, depth_mm)

#         # Generate YOLO segmentation labels and visualizations
#         labels_written = 0
#         with open(label_save_path, 'w') as f:
#             for obj_idx, obj in enumerate(sampled_target_bop_objs):
#                 # Find instance ID for this object
#                 obj_instance_id = None
#                 obj_name = obj.get_name()
                
#                 for inst_id, mapped_name in instance_name_map.items():
#                     if mapped_name == obj_name:
#                         obj_instance_id = inst_id
#                         break
                
#                 if obj_instance_id is None:
#                     continue
                
#                 # Extract mask for this object instance
#                 mask = (seg_mask_instances == obj_instance_id).astype(np.uint8)

#                 # Check if object is visible
#                 if not np.any(mask):
#                     continue

#                 # Convert mask to polygons
#                 polygons = mask_to_polygons(mask)

#                 if polygons:  # Only create visualization if we have valid polygons
#                     # Create mask visualization
#                     viz_name = f"scene{i:04d}_cam{frame_idx:02d}_obj{obj_idx:02d}_{obj_name}.jpg"
#                     viz_path = os.path.join(mask_viz_dir, viz_name)
                    
#                     # Convert polygons to normalized format for visualization
#                     norm_polygons = []
#                     for poly in polygons:
#                         norm_poly = []
#                         for k, p_coord in enumerate(poly):
#                             if k % 2 == 0:  # x coordinate
#                                 norm_poly.append(p_coord / img_width)
#                             else:  # y coordinate  
#                                 norm_poly.append(p_coord / img_height)
#                         norm_polygons.append(norm_poly)
                    
#                     visualize_mask_and_polygons(rgb_img, mask, norm_polygons, obj_name, viz_path)

#                 for poly in polygons:
#                     # Normalize polygon coordinates
#                     norm_poly = []
#                     for k, p_coord in enumerate(poly):
#                         if k % 2 == 0:  # x coordinate
#                             norm_poly.append(str(p_coord / img_width))
#                         else:  # y coordinate
#                             norm_poly.append(str(p_coord / img_height))

#                     class_id = CLASS_NAME_TO_ID["Lego_Block"]
#                     f.write(f"{class_id} " + " ".join(norm_poly) + "\n")
#                     labels_written += 1
  
#     # Clean up objects for next scene
#     for obj in sampled_target_bop_objs + sampled_distractor_bop_objs:
#         try:
#             obj.disable_rigidbody()
#             obj.hide(True)
#             obj.delete()
#         except Exception as e:
#             print(f"Warning: Error cleaning up object {obj.get_name()}: {e}")

#     print(f"Scene {i+1} completed successfully")
    
# print("Dataset generation complete.")

# # Generate dataset.yaml for YOLO
# dataset_yaml_content = {
#     'path': str(Path(output_dir).resolve()),
#     'train': 'images/train',
#     'val': 'images/val',
#     'nc': len(CLASS_NAME_TO_ID),
#     'names': list(CLASS_NAME_TO_ID.keys())
# }

# yaml_path = os.path.join(output_dir, 'dataset.yaml')
# with open(yaml_path, 'w') as f:
#     yaml.dump(dataset_yaml_content, f, default_flow_style=False)

# print(f"Generated dataset.yaml in {yaml_path}")

   
