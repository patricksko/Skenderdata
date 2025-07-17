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
import matplotlib.pyplot as plt
# Load YOLO model
model = YOLO("runs/Legoblock/weights/best.pt")

# Load image
img_path = "./prediction.jpg"
img = cv2.imread(img_path)

# Predict
results = model(
    source=img_path,
    imgsz=640,
    conf=0.5,
    save=True,   
    device="0"
)
# # Convert results to numpy
# boxes = results[0].boxes.xyxy.cpu().numpy()  # shape: [num_boxes, 4]

# for box in boxes:
#     xmin, ymin, xmax, ymax = box.astype(int)

#     crop = img[ymin:ymax, xmin:xmax]
    
#     # Split into B, G, R channels
#     colors = ('b', 'g', 'r')
#     plt.figure(figsize=(10, 4))

#     for i, col in enumerate(colors):
#         hist = cv2.calcHist([crop], [i], None, [256], [0, 256])
#         plt.plot(hist, color=col)
#         plt.xlim([0, 256])
#         plt.title("Color Histogram for Cropped Region")

#     plt.xlabel("Pixel Intensity")
#     plt.ylabel("Frequency")
#     plt.tight_layout()
#     plt.show()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#     # Show result (optional)
#     cv2.imshow("Rotated Bounding Box", thresh)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # Find contours
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter by area and get largest contour
#     contours = sorted(contours, key=cv2.contourArea, reverse=True)
#     largest = contours[0]

#     # Get the minimum-area rotated rectangle
#     rect = cv2.minAreaRect(largest)
#     box = cv2.boxPoints(rect)
#     box = np.intp(box)

#     # Draw the rotated bounding box
#     rotated = img.copy()
#     cv2.drawContours(rotated, [box], 0, (0, 255, 0), 2)

#     # Show result (optional)
#     cv2.imshow("Rotated Bounding Box", rotated)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

