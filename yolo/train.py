from ultralytics import YOLO
 
model = YOLO("yolo11n.pt")

result = model.train(
    data="./examples/my_examples/data.yaml",
    epochs = 100,
    imgsz=640,
    batch=16,
    optimizer="Adam",
    lr0=0.001,
    device="0",
    name="Lego_Block",
    save=True,
    save_json=True,
    project="runs",
    exist_ok=True,
    resume=False,
    patience=10,
)
# results = model("./examples/my_examples/Lego_Block.jpeg", conf=0.25)
# for r in results:
#     r.save(filename="./prediction.jpg")