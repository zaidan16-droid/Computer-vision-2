import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_objects(image):
    results = model(image)
    return results.render()[0]  # gambar dengan bounding box
