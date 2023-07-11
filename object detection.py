from ultralytics import YOLO

model = YOLO("yolov8n.yaml")

result = model.train(data="parshva.yaml", epochs=1)