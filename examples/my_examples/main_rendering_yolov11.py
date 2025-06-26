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
import yaml
import random
import debugpy
# from render_funcs import sample_pose_func, mask_to_polygons, visualize_mask_and_polygons

# debugpy.listen(5678)
# debugpy.wait_for_client()


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






######################################################################################################################






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

# Load BOP objects - check if paths exist before loading
target_bop_objs = []
if os.path.exists(os.path.join(bop_dataset_path, 'Legoblock')):
    target_bop_objs = bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(bop_dataset_path, 'Legoblock'), 
        object_model_unit='mm', 
        obj_ids=[1, 2, 3]
    )
    print(f"Loaded {len(target_bop_objs)} target objects")
else:
    print(f"Warning: Legoblock path {os.path.join(bop_dataset_path, 'Legoblock')} not found")

# Load distractor objects (with error handling)
tless_dist_bop_objs = []
if os.path.exists(os.path.join(bop_dataset_path, 'tless')):
    try:
        tless_dist_bop_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(bop_dataset_path, 'tless'), 
            model_type='cad', 
            object_model_unit='mm'
        )
        print(f"Loaded {len(tless_dist_bop_objs)} T-LESS objects")
    except Exception as e:
        print(f"Warning: Could not load T-LESS objects: {e}")

hb_dist_bop_objs = []
if os.path.exists(os.path.join(bop_dataset_path, 'hb')):
    try:
        hb_dist_bop_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(bop_dataset_path, 'hb'), 
            object_model_unit='mm'
        )
        print(f"Loaded {len(hb_dist_bop_objs)} HB objects")
    except Exception as e:
        print(f"Warning: Could not load HB objects: {e}")

tyol_dist_bop_objs = []
if os.path.exists(os.path.join(bop_dataset_path, 'tyol')):
    try:
        tyol_dist_bop_objs = bproc.loader.load_bop_objs(
            bop_dataset_path=os.path.join(bop_dataset_path, 'tyol'), 
            object_model_unit='mm'
        )
        print(f"Loaded {len(tyol_dist_bop_objs)} TYOL objects")
    except Exception as e:
        print(f"Warning: Could not load TYOL objects: {e}")

print("Dataloading accomplished")

# Load intrinsics (check if path exists)
if os.path.exists(os.path.join(bop_dataset_path, 'ycbv')):
    bproc.loader.load_bop_intrinsics(bop_dataset_path=os.path.join(bop_dataset_path, 'ycbv'))
    print("Loaded YCBV intrinsics")
else:
    print("Warning: YCBV intrinsics not found, using default camera settings")

# Set shading and hide objects initially
all_objects = target_bop_objs + tless_dist_bop_objs + hb_dist_bop_objs + tyol_dist_bop_objs
for obj in all_objects:
    obj.set_shading_mode('auto')
    obj.hide(True)

# Create room
room_planes = [
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
    bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])
]

for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

# Lights
light_plane = bproc.object.create_primitive('PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')
light_point = bproc.types.Light()

# Load textures
cc_textures = []
if os.path.exists(cc_textures_path):
    try:
        cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
        print(f"Loaded {len(cc_textures)} CC textures")
    except Exception as e:
        print(f"Warning: Could not load CC textures: {e}")
else:
    print(f"Warning: CC textures path {cc_textures_path} not found")

# Load custom textures
custom_texture_dir = './resources/my_textures/'
custom_texture_mats = []
if os.path.exists(custom_texture_dir):
    image_files = [f for f in os.listdir(custom_texture_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if image_files:
        for i, filename in enumerate(image_files):
            try:
                tex_path = os.path.join(custom_texture_dir, filename)
                mat = bproc.material.create_material_from_texture(tex_path, f"custom_tex_{i}")
                custom_texture_mats.append(mat)
            except Exception as e:
                print(f"Warning: Could not load texture {filename}: {e}")
        print(f"Loaded {len(custom_texture_mats)} custom textures")
    else:
        print(f"Warning: No image files found in '{custom_texture_dir}'")
else:
    print(f"Warning: Custom texture directory '{custom_texture_dir}' not found")


# Enable segmentation output
bproc.renderer.enable_segmentation_output(map_by=["instance", "name"])
bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.set_max_amount_of_samples(50)



print("Starting scene generation...")

global_object_counter = 0

for i in range(num_scenes):
    print(f"Generating scene {i+1}/{num_scenes}")
    
    # Skip if no target objects available
    if not target_bop_objs:
        print("No target objects available, skipping scene generation")
        break
    
    # Generate random colors
    colors = [[*np.random.rand(3), 1.0] for _ in range(6)]

    # Sample number of objects for this scene
    obj_config = json_config.get('scene', {}).get('objects', {'min_count': 1, 'max_count': 5})
    num_target_objs = np.random.randint(obj_config['min_count'], obj_config['max_count'] + 1)
    num_distractor_tless = np.random.randint(0, min(3, len(tless_dist_bop_objs)) + 1) if tless_dist_bop_objs else 0
    num_distractor_hb = np.random.randint(0, min(3, len(hb_dist_bop_objs)) + 1) if hb_dist_bop_objs else 0
    num_distractor_tyol = np.random.randint(0, min(3, len(tyol_dist_bop_objs)) + 1) if tyol_dist_bop_objs else 0

    # Sample target objects with proper copying
    sampled_target_bop_objs = []
    for _ in range(num_target_objs):
        original_obj = random.choice(target_bop_objs)
        # Create a copy of the object
        copied_obj = original_obj.duplicate()
        copied_obj.set_name(f"target_obj_{global_object_counter}")
        global_object_counter += 1
        sampled_target_bop_objs.append(copied_obj)

    # Sample distractor objects with proper copying
    sampled_distractor_bop_objs = []
    
    for dist_objs, num_dist, prefix in [
        (tless_dist_bop_objs, num_distractor_tless, "tless"),
        (hb_dist_bop_objs, num_distractor_hb, "hb"),
        (tyol_dist_bop_objs, num_distractor_tyol, "tyol")
    ]:
        if dist_objs and num_dist > 0:
            for _ in range(num_dist):
                original_obj = random.choice(dist_objs)
                copied_obj = original_obj.duplicate()
                copied_obj.set_name(f"distractor_{prefix}_obj_{global_object_counter}")
                global_object_counter += 1
                sampled_distractor_bop_objs.append(copied_obj)

    all_sampled_objects = sampled_target_bop_objs + sampled_distractor_bop_objs
    
    if not all_sampled_objects:
        print(f"Warning: No objects to sample for scene {i+1}. Skipping...")
        continue

    print(f"Scene {i+1}: {len(sampled_target_bop_objs)} target objects, {len(sampled_distractor_bop_objs)} distractor objects")

    # Apply materials and physics
    for obj in all_sampled_objects:
        obj.set_shading_mode('auto')
        obj.hide(False)

        # Apply materials
        if obj in sampled_target_bop_objs:
            # Target objects: random texture or color
            if random.random() < 0.2 and custom_texture_mats:
                random_texture = random.choice(custom_texture_mats)
                # Create a copy of the material to avoid conflicts
                mat_copy = random_texture.duplicate()
                obj.replace_materials(mat_copy)
                mat_copy.set_principled_shader_value("Roughness", np.random.uniform(0.2, 0.9))
                mat_copy.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.0, 0.5))
            else:
                # Create new material for this object
                mat = bproc.material.create(f"target_mat_{global_object_counter}")
                obj.replace_materials(mat)
                mat.set_principled_shader_value("Base Color", random.choice(colors))
                mat.set_principled_shader_value("Roughness", np.random.uniform(0.5, 0.9))
                mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.1, 0.6))
        else:
            # Distractor objects: grey color
            mat = bproc.material.create(f"distractor_mat_{global_object_counter}")
            obj.replace_materials(mat)
            grey_col = np.random.uniform(0.1, 0.9)
            mat.set_principled_shader_value("Base Color", [grey_col, grey_col, grey_col, 1])
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.0, 1.0))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.0, 1.0))

        obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)

    # Set lighting
    light_config = json_config.get('scene', {}).get('lights', {'min_intensity': 3, 'max_intensity': 6})
    lp_intensity = np.random.uniform(light_config['min_intensity'], light_config['max_intensity'])
    light_point.set_energy(lp_intensity)
    
    light_plane_material.make_emissive(
        emission_strength=np.random.uniform(3, 6),
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0])
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5, elevation_min=5, elevation_max=89)
    light_point.set_location(location)

    # Apply background texture to room planes
    if cc_textures:
        random_cc_texture = random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(random_cc_texture)

    # Sample poses and simulate physics
    try:
        bproc.object.sample_poses(
            objects_to_sample=all_sampled_objects,
            sample_pose_func=sample_pose_func,
            max_tries=1000
        )
        
        bproc.object.simulate_physics_and_fix_final_poses(
            min_simulation_time=3,
            max_simulation_time=10,
            check_object_interval=1,
            substeps_per_frame=20,
            solver_iters=25
        )
    except Exception as e:
        print(f"Warning: Physics simulation failed for scene {i+1}: {e}")
        # Clean up and continue to next scene
        for obj in all_sampled_objects:
            try:
                obj.disable_rigidbody()
                obj.hide(True)
                obj.delete()
            except:
                pass
        continue

    # Create BVH tree for camera pose sampling
    try:
        bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(all_sampled_objects)
    except Exception as e:
        print(f"Warning: Could not create BVH tree for scene {i+1}: {e}")
        # Clean up and continue
        for obj in all_sampled_objects:
            try:
                obj.disable_rigidbody()
                obj.hide(True)
                obj.delete()
            except:
                pass
        continue

    # Sample camera poses
    cam_poses_generated = 0
    max_camera_tries = 1000
    camera_tries = 0
    
    while cam_poses_generated < num_cameras and camera_tries < max_camera_tries:
        camera_tries += 1
        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=0.61, radius_max=1.24, elevation_min=5, elevation_max=89)
        
        # Point camera towards target objects if available
        if sampled_target_bop_objs:
            num_poi_objs = min(len(sampled_target_bop_objs), 5)
            random_poi_selection = np.random.choice(sampled_target_bop_objs, size=num_poi_objs, replace=False)
            poi = bproc.object.compute_poi(random_poi_selection)
        else:
            poi = np.array([0, 0, 0])

        rotation_matrix = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=np.random.uniform(-np.pi, np.pi))
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)

        # Perform obstacle check
        try:
            if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
                bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses_generated)
                cam_poses_generated += 1
        except Exception as e:
            # If obstacle check fails, skip this camera pose
            continue

    if cam_poses_generated == 0:
        print(f"Warning: No valid camera poses found for scene {i+1}. Skipping...")
        # Clean up objects
        for obj in all_sampled_objects:
            try:
                obj.disable_rigidbody()
                obj.hide(True)
                obj.delete()
            except:
                pass
        continue

    print(f"Generated {cam_poses_generated} camera poses for scene {i+1}")

    # Render scene
    try:
        data = bproc.renderer.render()
    except Exception as e:
        print(f"Warning: Rendering failed for scene {i+1}: {e}")
        # Clean up and continue
        for obj in all_sampled_objects:
            try:
                obj.disable_rigidbody()
                obj.hide(True)
                obj.delete()
            except:
                pass
        continue

    # Check if rendering was successful
    if not data.get("colors") or len(data["colors"]) == 0:
        print(f"Warning: No images rendered for scene {i+1}. Skipping label generation.")
        # Clean up objects
        for obj in all_sampled_objects:
            try:
                obj.disable_rigidbody()
                obj.hide(True)
                obj.delete()
            except:
                pass
        continue

    # Get image dimensions
    img_height, img_width, _ = data["colors"][0].shape
    print(f"Rendered {len(data['colors'])} images for scene {i+1} ({img_width}x{img_height})")

    # Process each rendered frame
    for frame_idx in range(len(data["colors"])):
        rgb_img = (data["colors"][frame_idx] * 255).astype(np.uint8)

        # Get segmentation data
        seg_mask_instances = data["instance_segmaps"][frame_idx]
        
        # Build instance name mapping
        instance_name_map = {}
        if "instance_attribute_maps" in data and len(data["instance_attribute_maps"]) > frame_idx:
            for item in data["instance_attribute_maps"][frame_idx]:
                instance_name_map[item["idx"]] = item["name"]

        image_name = f"scene{i:04d}_cam{frame_idx:02d}.jpg"

        # Determine split (train/val)
        if np.random.rand() < args.val_split_ratio:
            split = 'val'
            image_save_path = os.path.join(image_val_dir, image_name)
            label_save_path = os.path.join(label_seg_val_dir, image_name.replace('.jpg', '.txt'))
        else:
            split = 'train'
            image_save_path = os.path.join(image_train_dir, image_name)
            label_save_path = os.path.join(label_seg_train_dir, image_name.replace('.jpg', '.txt'))

        # Save image
        Image.fromarray(rgb_img).save(image_save_path)

        # Generate YOLO segmentation labels and visualizations
        labels_written = 0
        with open(label_save_path, 'w') as f:
            for obj_idx, obj in enumerate(sampled_target_bop_objs):
                # Find instance ID for this object
                obj_instance_id = None
                obj_name = obj.get_name()
                
                for inst_id, mapped_name in instance_name_map.items():
                    if mapped_name == obj_name:
                        obj_instance_id = inst_id
                        break
                
                if obj_instance_id is None:
                    continue
                
                # Extract mask for this object instance
                mask = (seg_mask_instances == obj_instance_id).astype(np.uint8)

                # Check if object is visible
                if not np.any(mask):
                    continue

                # Convert mask to polygons
                polygons = mask_to_polygons(mask)

                if polygons:  # Only create visualization if we have valid polygons
                    # Create mask visualization
                    viz_name = f"scene{i:04d}_cam{frame_idx:02d}_obj{obj_idx:02d}_{obj_name}.jpg"
                    viz_path = os.path.join(mask_viz_dir, viz_name)
                    
                    # Convert polygons to normalized format for visualization
                    norm_polygons = []
                    for poly in polygons:
                        norm_poly = []
                        for k, p_coord in enumerate(poly):
                            if k % 2 == 0:  # x coordinate
                                norm_poly.append(p_coord / img_width)
                            else:  # y coordinate  
                                norm_poly.append(p_coord / img_height)
                        norm_polygons.append(norm_poly)
                    
                    visualize_mask_and_polygons(rgb_img, mask, norm_polygons, obj_name, viz_path)

                for poly in polygons:
                    # Normalize polygon coordinates
                    norm_poly = []
                    for k, p_coord in enumerate(poly):
                        if k % 2 == 0:  # x coordinate
                            norm_poly.append(str(p_coord / img_width))
                        else:  # y coordinate
                            norm_poly.append(str(p_coord / img_height))

                    class_id = CLASS_NAME_TO_ID["Lego_Block"]
                    f.write(f"{class_id} " + " ".join(norm_poly) + "\n")
                    labels_written += 1

        print(f"  Frame {frame_idx}: {labels_written} labels written")

        # Draw bounding boxes for debugging if enabled
        if json_config.get("draw_bbox", False):
            img_with_bbox = rgb_img.copy()
            for obj in sampled_target_bop_objs:
                try:
                    points = bproc.camera.project_points(obj.get_bound_box(), frame=frame_idx)
                    
                    if len(points) > 0:
                        min_xy = np.min(points, axis=0).astype(int)
                        max_xy = np.max(points, axis=0).astype(int)
                        min_x, min_y = max(0, min_xy[0]), max(0, min_xy[1])
                        max_x, max_y = min(img_width - 1, max_xy[0]), min(img_height - 1, max_xy[1])

                        if max_x > min_x and max_y > min_y:
                            cv2.rectangle(img_with_bbox, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Warning: Could not draw bbox for object {obj.get_name()}: {e}")

            Image.fromarray(img_with_bbox).save(os.path.join(bbox_overlay_dir, image_name))

    # Clean up objects for next scene
    for obj in all_sampled_objects:
        try:
            obj.disable_rigidbody()
            obj.hide(True)
            obj.delete()
        except Exception as e:
            print(f"Warning: Error cleaning up object {obj.get_name()}: {e}")

    print(f"Scene {i+1} completed successfully")

print("Dataset generation complete.")

# Generate dataset.yaml for YOLO
dataset_yaml_content = {
    'path': str(Path(output_dir).resolve()),
    'train': 'images/train',
    'val': 'images/val',
    'nc': len(CLASS_NAME_TO_ID),
    'names': list(CLASS_NAME_TO_ID.keys())
}

yaml_path = os.path.join(output_dir, 'dataset.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_yaml_content, f, default_flow_style=False)

print(f"Generated dataset.yaml in {yaml_path}")