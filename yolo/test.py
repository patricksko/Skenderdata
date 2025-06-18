from ultralytics import YOLO
from pathlib import Path
import os

rel_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path= os.path.join(rel_path, "examples/my_examples/output_blenderproc/bbox_overlay")
print(os.path.exists(config_path))

model = YOLO("yolo11n.pt")  # initialize model
results = model(os.path.join(config_path, "0000_07.jpg"))  # perform inference
results[0].show()  # display results for the first image

#/../examples/my_examples/output_blenderproc/bbox_overlay