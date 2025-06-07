from ultralytics import YOLO
def load_model():
    model = YOLO("yolov8n.pt")  # model kecil
    return model
