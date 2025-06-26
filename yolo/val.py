from ultralytics import YOLO
 
model = YOLO("./runs/Lego_Block/weights/last.pt")

metrics = model.val(
    data="./examples/my_examples/output_blenderproc/data.yaml",
    split="val",
    imgsz=640,
    conf=0.9,
    iou=0.6,
    device="0",
    save=True
)
print(metrics.results_dict)