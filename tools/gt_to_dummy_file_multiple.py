import json
import os

# Path to the root folder containing the subdirectories
root_folder = '../BOP/gdrnpp_bop2022/datasets/BOP_DATASETS/ycb_ichores/test'  # Replace with the actual path

output_data = {}

# Iterate through subdirectories
for subdir in sorted(os.listdir(root_folder)):
    subfolder_path = os.path.join(root_folder, subdir)
    
    if os.path.isdir(subfolder_path):
        scene_number = int(subdir)  # Convert subdir to an integer
        
        scene_gt_info_path = os.path.join(subfolder_path, 'scene_gt_info.json')
        scene_gt_path = os.path.join(subfolder_path, 'scene_gt.json')

        if os.path.exists(scene_gt_info_path):
            # Read the JSON file
            with open(scene_gt_info_path) as file:
                data = json.load(file)
            with open(scene_gt_path) as file:
                data_gt = json.load(file)

            # Process the data
            for key, items in data.items():
                image_id = int(key)  # Convert key to an integer
                obj_key = f"{scene_number}/{image_id}"  # Correct key format for scene/image_id
                
                items_gt = data_gt[key]
                output_items = []
                
                for index, item in enumerate(items):
                    obj_id =items_gt[index]['obj_id']
                    bbox_obj = item['bbox_obj']
                    score = 1.0  # Placeholder value, you can replace it with the actual score
                    time = 0.2  # Placeholder value, you can replace it with the actual time

                    # Create the output item
                    output_item = {
                        'obj_id': obj_id,
                        'bbox_est': bbox_obj,
                        'score': score,
                        'time': time
                    }

                    # Append the item to the output items list
                    output_items.append(output_item)

                # Add the items list to the output data
                output_data[obj_key] = output_items

# Write the output JSON file
with open('scene_gt_dummy_bb.json', 'w') as file:
    json.dump(output_data, file, indent=2)
