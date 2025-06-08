import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from detector import ObjectCounter

st.set_page_config(page_title="Object Counter", layout="wide")
st.title("🔢 Object Counter via WebRTC")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = ObjectCounter()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = self.counter.detect_and_count(img)
        return img

webrtc_streamer(
    key="object-counter",
    video_processor_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
)
