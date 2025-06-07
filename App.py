from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # model kecil
    return model
