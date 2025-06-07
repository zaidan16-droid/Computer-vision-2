import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from utils.detection import detect_objects

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        output_img = detect_objects(img)  # proses deteksi
        return av.VideoFrame.from_ndarray(output_img, format="bgr24")

st.title("Object Detection with WebRTC")
webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
