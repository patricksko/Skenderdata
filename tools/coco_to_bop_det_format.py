import json
import argparse

def convert_json(input_path, output_path):
    with open(input_path, 'r') as json_file:
        data = json.load(json_file)
        
    converted_data = {}
    
    for entry in data:
        scene_id = entry['scene_id']
        image_id = entry['image_id']
        obj_id = entry['category_id']
        bbox_est = entry['bbox']
        score = entry['score']
        time = entry['time']
        
        key = f"{scene_id}/{image_id}"
        if key not in converted_data:
            converted_data[key] = []
        
        converted_entry = {
            "obj_id": obj_id,
            "bbox_est": bbox_est,
            "score": score,
            "time": time
        }
        
        converted_data[key].append(converted_entry)
    
    with open(output_path, 'w') as output_file:
        json.dump(converted_data, output_file, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert JSON format')
    parser.add_argument('input', help='Path to the input JSON file')
    parser.add_argument('output', help='Path to the output JSON file')
    args = parser.parse_args()

    convert_json(args.input, args.output)
