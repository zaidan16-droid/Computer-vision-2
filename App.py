import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
from utils.detection import detect_objects

class VideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = detect_objects(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🧠 Real-Time Object Detection")
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
