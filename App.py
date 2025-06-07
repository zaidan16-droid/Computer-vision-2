import streamlit as st
from streamlit-webrtc import webrtc-streamer, VideoTransformerBase
import av
import torch
import numpy as np

# Load YOLOv5 model from torch.hub
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    return model

model = load_model()

class ObjectDetection(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img)
        result_img = np.squeeze(results.render())  # tambahkan bounding box
        return av.VideoFrame.from_ndarray(result_img, format="bgr24")

st.title("🧠 Real-Time Object Detection with YOLOv5")

webrtc_streamer(
    key="yolov5-detect",
    video_processor_factory=ObjectDetection,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
