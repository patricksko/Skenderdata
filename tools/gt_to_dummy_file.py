import json

# Read the JSON file
with open('test/000002/scene_gt_info.json') as file:
    data = json.load(file)
with open('test/000002/scene_gt.json') as file:
    data_gt = json.load(file)

scene_number = 2
output_data = {}

# Process the data
for key, items in data.items():
    items_gt = data_gt[key]

    for index, item in enumerate(items):
        image_id = int(key)  # Convert key to an integer
        obj_key = f"{scene_number}/{image_id}"  # Correct key format for scene/image_id
        
        items_gt = data_gt[key]

        for index, item in enumerate(items):
            obj_id =items_gt[index]['obj_id']
            bbox_obj = item['bbox_obj']
            score = 1.0  # Placeholder value, you can replace it with the actual score
            time = 0.0  # Placeholder value, you can replace it with the actual time

            # Create the output item
            output_item = {
                'obj_id': obj_id,
                'bbox_est': bbox_obj,
                'score': score,
                'time': time
            }

        # Add the item to the output data
        if obj_key in output_data:
            output_data[obj_key].append(output_item)
        else:
            output_data[obj_key] = [output_item]

# Write the output JSON file
with open('gt_bb.json', 'w') as file:
    json.dump(output_data, file, indent=2)
