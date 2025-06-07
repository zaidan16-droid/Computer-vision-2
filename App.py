import streamlit as st
from detect_utils import detect_objects

st.title("🔍 Real-Time Object Detection")

# Konfigurasi WebRTC (opsional, bisa dikosongkan juga)
rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = detect_objects(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
)
