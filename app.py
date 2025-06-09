
import cv2
import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load model YOLOv8
model = YOLO("yolov8n.pt")

class YOLOVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        annotated = results[0].plot()
        return annotated

st.set_page_config(page_title="YOLOv8 Real-Time")
st.title("🎥 Deteksi Objek Real-Time (Webcam + YOLOv8)")

webrtc_streamer(key="yolo", video_transformer_factory=YOLOVideoTransformer)
