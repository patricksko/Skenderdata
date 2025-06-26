# from ultralytics import YOLO

# # Load your trained model
# model = YOLO("runs/Lego_Block/weights/best.pt")

# # Predict on your own images (no labels needed)
# results = model.predict(
#     source="./examples/my_examples/Lego_Block1.jpeg",  # can be a folder, single image, or video
#     imgsz=640,
#     conf=0.5,   # optional, confidence threshold
#     save=True,  # optional, saves result images to runs/predict
#     device="0"  # or "cpu" if no GPU
# )
import cv2
import numpy as np
from ultralytics import YOLO
# Load YOLO model
model = YOLO("runs/Lego_Block/weights/best.pt")

# Load image
img_path = "./examples/my_examples/Lego_Block2.jpeg"
img = cv2.imread(img_path)

# Predict
results = model.predict(
    source=img_path,
    imgsz=640,
    conf=0.5,
    save=False,   # We'll handle visualization ourselves here
    device="0"
)
# Convert results to numpy
boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: [num_boxes, 4]

for box in boxes:
    xmin, ymin, xmax, ymax = box.astype(int)

    # Crop the detected region
    crop = img[ymin:ymax, xmin:xmax]

    # Convert crop to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Threshold the image to get binary mask
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue  # no contours found, skip

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Ignore small contours that may be noise
    if cv2.contourArea(largest_contour) < 20:
        continue

    # Compute the minimum area rotated rectangle
    rect = cv2.minAreaRect(largest_contour)
    box_points = cv2.boxPoints(rect)
    box_points = np.int0(box_points)

    # Since box_points are relative to crop, shift them to original image coords
    box_points[:, 0] += xmin
    box_points[:, 1] += ymin

    # Draw oriented bounding box on original image
    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)

# Show the final image with oriented boxes
cv2.imshow("Oriented Bounding Boxes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
