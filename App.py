import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from detector import ObjectCounter

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.counter = ObjectCounter()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, count = self.counter.detect_and_count(img)
        return img

st.title("🔢 Object Counter from Webcam")

webrtc_streamer(key="object-counter", video_processor_factory=VideoTransformer)
