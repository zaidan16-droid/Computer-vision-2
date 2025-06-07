import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import torch
import numpy as np

# Load model YOLOv5 dari PyTorch Hub (bebas dari ukuran file lokal)
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

# Kelas untuk memproses video frame demi frame
class ObjectDetection(VideoTransformerBase):
    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Deteksi objek dengan YOLO
        results = model(image)
        result_image = np.squeeze(results.render())  # gambar output dengan bounding box

        return av.VideoFrame.from_ndarray(result_image, format="bgr24")

# Judul aplikasi
st.title("🧠 Real-Time Object Detection with YOLOv5 + Streamlit WebRTC")

# WebRTC Streamer
webrtc_streamer(
    key="object-detection",
    video_processor_factory=ObjectDetection,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
