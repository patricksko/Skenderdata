import os
import json

def get_targets_from_file(target_file):
    with open(target_file, 'r') as json_file:
        data = json.load(json_file)

    # Create a dictionary to store unique im_ids for each scene_id
    scene_id_to_im_ids = {}

    for item in data:
        im_id = item['im_id']
        scene_id = item['scene_id']
        
        # Check if the combination of im_id and scene_id is unique
        if scene_id not in scene_id_to_im_ids:
            scene_id_to_im_ids[scene_id] = set()  # Use a set to store unique im_ids
        
        # Add the im_id to the set for the corresponding scene_id
        scene_id_to_im_ids[scene_id].add(im_id)

    # Sort the unique im_ids for each scene_id in ascending order
    for scene_id in scene_id_to_im_ids:
        scene_id_to_im_ids[scene_id] = sorted(scene_id_to_im_ids[scene_id])

    return scene_id_to_im_ids

def modify_json_files(input_folder, output_folder, dataset, target_file):
    folder_names = sorted(os.listdir(input_folder), key=lambda x: int(x))

    if target_file is not None:
        targets = get_targets_from_file(target_file)

    for folder_name in folder_names:
        folder_path = os.path.join(input_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        scene_id = int(folder_name)
        modified_annotations = {}

        with open(os.path.join(folder_path, "scene_gt_coco.json"), "r") as json_file:
            data = json.load(json_file)

            valid_category_ids = [1, 5, 6, 8, 9, 10, 11, 12]
            category_id_mapping = {
                1: 1,
                5: 2,
                6: 3,
                8: 4,
                9: 5,
                10: 6,
                11: 7,
                12: 8
            }

            to_remove = []
            for image_data in data["categories"]:
                if dataset == "lmo" and image_data["id"] not in valid_category_ids:
                    to_remove.append(image_data)
                elif dataset == "lmo" and image_data["id"] in category_id_mapping:
                    image_data["name"] = str(category_id_mapping[image_data["id"]])
                    image_data["id"] = category_id_mapping[image_data["id"]]

            for item in to_remove:
                data["categories"].remove(item)
            
            to_remove = []

            # Update images file names and IDs
            for image_data in data["images"]:
                if target_file is not None:
                    if image_data["id"] not in targets[scene_id]:
                        to_remove.append(image_data)
                    else:
                        image_data["file_name"] = f"{folder_name}/{image_data['file_name']}"
                        image_data["id"] += scene_id * 1000  # Assuming each scene has less than 1000 images                        
                else:
                    image_data["file_name"] = f"{folder_name}/{image_data['file_name']}"
                    image_data["id"] += scene_id * 1000  # Assuming each scene has less than 1000 images

            for item in to_remove:
                data["images"].remove(item)

            # Update images file names and IDs
            to_remove = []

            for image_data in data["annotations"]:
                if target_file is not None:
                    if image_data["image_id"] not in targets[scene_id]:
                        to_remove.append(image_data)
                    else:
                        image_data["image_id"] += scene_id * 1000  # Assuming each scene has less than 1000 images
                        image_data["id"] += scene_id * 100000
                        if (dataset == "lmo" and image_data["category_id"] not in valid_category_ids) or image_data["ignore"] == "true":
                            to_remove.append(image_data)
                        elif dataset == "lmo" and image_data["category_id"] in category_id_mapping:
                            image_data["category_id"] = category_id_mapping[image_data["category_id"]]
                        del image_data["ignore"]
                else:
                    image_data["image_id"] += scene_id * 1000  # Assuming each scene has less than 1000 images
                    image_data["id"] += scene_id * 100000
                    if (dataset == "lmo" and image_data["category_id"] not in valid_category_ids) or image_data["ignore"] == "true":
                        to_remove.append(image_data)
                    elif dataset == "lmo" and image_data["category_id"] in category_id_mapping:
                        image_data["category_id"] = category_id_mapping[image_data["category_id"]]
                    del image_data["ignore"]

            for item in to_remove:
                data["annotations"].remove(item)

            modified_annotations = data

        # Save the modified JSON file to the output folder
        output_file_path = os.path.join(output_folder, f"scene_gt_coco_{folder_name}.json")
        with open(output_file_path, "w") as output_file:
            json.dump(modified_annotations, output_file)

def combine_json_files(input_folder, output_file):
    combined_data = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }

    first_file = True
    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".json"):
            continue

        file_path = os.path.join(input_folder, file_name)
        with open(file_path, "r") as json_file:
            data = json.load(json_file)

            # Load metadata only from the first file
            if first_file:
                combined_data["info"] = data["info"]
                combined_data["licenses"] = data["licenses"]
                combined_data["categories"] = data["categories"]
                first_file = False

            # Combine images and annotations information
            combined_data["images"].extend(data["images"])
            combined_data["annotations"].extend(data["annotations"])

    # Save the combined JSON file
    with open(output_file, "w") as output_json:
        json.dump(combined_data, output_json)

if __name__ == "__main__":
    #input_folder = "../temp/detectron2_bop/datasets/BOP_DATASETS/tless_random_texture/train_pbr"
    input_folder = "/hdd/datasets_bop/mp6d_random_texture/train_pbr"
    output_folder = "./modified_annotations"
    dataset = "mp6d"
    #target_file = "../temp/mmdetection/data/tless/test_targets_bop19.json"
    target_file = None

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    modify_json_files(input_folder, output_folder, dataset, target_file)
    combine_json_files(output_folder, output_folder + "/" + "combined.json")
