import os
import numpy as np
import cv2
from PIL import Image
import debugpy


# debugpy.listen(5678)
# debugpy.wait_for_client()

# base paths

camera_info = {
  "cx": 312.9869,
  "cy": 241.3109,
  "depth_scale": 0.1,
  "fx": 1066.778,
  "fy": 1067.487,
"height": 480,
  "width": 640
}


dataset_root = "my_examples/output_blenderproc"
split = "train"  # or "val"

images_dir = os.path.join(dataset_root, "images", split)
depth_dir = os.path.join(dataset_root, "depth", split)
labels_dir = os.path.join(dataset_root, "labels", split)

# list all images
image_files = sorted(os.listdir(images_dir))

for img_name in image_files:
    img_path = os.path.join(images_dir, img_name)
    depth_path = os.path.join(depth_dir, img_name.replace(".jpg", ".png"))
    label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))

    # load RGB image
    rgb = cv2.imread(img_path)
    h, w, _ = rgb.shape

    # load depth (16-bit or float)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # initialize blank mask
    instance_mask = np.zeros((h, w), dtype=np.uint8)

    # parse YOLO polygon segmentation
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parts = line.strip().split()
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))

            # YOLO segmentation uses normalized coords
            xy_points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * w
                y = coords[i+1] * h
                xy_points.append([x, y])

            xy_points = np.array(xy_points, np.int32)
            cv2.fillPoly(instance_mask, [xy_points], color=idx+1)  # +1 to distinguish from background

    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances != 0]

    fx, fy = camera_info["fx"], camera_info["fy"]
    cx, cy = camera_info["cx"], camera_info["cy"]
    depth_scale = camera_info["depth_scale"]

    for instance_id in unique_instances:
        ys, xs = np.where(instance_mask == instance_id)

        u_center = int(np.mean(xs))
        v_center = int(np.mean(ys))

        z_raw = depth[v_center, u_center]
        z = z_raw * depth_scale  # scale to meters

        # Backproject to 3D
        X = (u_center - cx) * z / fx
        Y = (v_center - cy) * z / fy
        Z = z

        print(f"Instance {instance_id} (polygon {instance_id - 1}):")
        print(f"  Pixel center (u,v): ({u_center}, {v_center})")
        print(f"  Depth (raw, scaled): {z_raw}, {z:.3f} m")
        print(f"  3D coordinates (X,Y,Z): ({X:.3f}, {Y:.3f}, {Z:.3f}) meters")



    # optionally visualize
    cv2.imshow("rgb", rgb)
    cv2.imshow("depth", (depth / np.max(depth)).astype(np.float32))
    cv2.imshow("mask", (instance_mask * 50).astype(np.uint8))
    cv2.waitKey(0)

cv2.destroyAllWindows()