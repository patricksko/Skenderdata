import os
import yaml
import argparse

def export_yolo_yaml(dataset_root, train_split=0.9, class_name_to_id=None, output_yaml_path="data.yaml"):
    assert class_name_to_id is not None, "You must provide class_name_to_id dictionary."

    # Paths to images
    image_dir = os.path.join(dataset_root, "images")
    label_dir = os.path.join(dataset_root, "labels")

    images = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
    num_train = int(len(images) * train_split)
    train_imgs = images[:num_train]
    val_imgs = images[num_train:]

    # Write file lists for YOLO
    def write_split_list(split_imgs, split_name):
        split_path = os.path.join(dataset_root, f"{split_name}.txt")
        with open(split_path, "w") as f:
            for img_name in split_imgs:
                full_path = os.path.abspath(os.path.join(image_dir, img_name))
                f.write(full_path + "\n")
        return split_path

    train_txt = write_split_list(train_imgs, "train")
    val_txt = write_split_list(val_imgs, "val")

    # Build YAML content
    yaml_data = {
        "train": train_txt,
        "val": val_txt,
        "nc": len(class_name_to_id),
        "names": [None] * len(class_name_to_id)
    }

    for name, idx in class_name_to_id.items():
        yaml_data["names"][idx] = name

    with open(output_yaml_path, "w") as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)

    print(f"YAML config written to: {output_yaml_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO YAML dataset description")
    parser.add_argument("--dataset_root", required=True, help="Path to the dataset output folder")
    parser.add_argument("--split", type=float, default=0.9, help="Train/val split ratio (default 0.9)")
    parser.add_argument("--output", type=str, default="data.yaml", help="Output YAML path")
    args = parser.parse_args()

    # Change this to your class dictionary
    CLASS_NAME_TO_ID = {
        "Lego_Block": 0
    }

    export_yolo_yaml(args.dataset_root, train_split=args.split,
                     class_name_to_id=CLASS_NAME_TO_ID, output_yaml_path=args.output)
