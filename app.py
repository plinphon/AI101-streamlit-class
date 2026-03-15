import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="AI101 Vision",
    page_icon="🤖",
    layout="wide"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>

.stApp{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

.big-title{
font-size:60px;
font-weight:800;
text-align:center;
margin-bottom:10px;
}

.subtitle{
text-align:center;
font-size:20px;
opacity:0.8;
margin-bottom:40px;
}

.card{
background: rgba(255,255,255,0.1);
padding:20px;
border-radius:20px;
backdrop-filter: blur(10px);
box-shadow:0px 10px 30px rgba(0,0,0,0.4);
}

</style>
""", unsafe_allow_html=True)

# ---------- HERO ----------
st.markdown('<div class="big-title">AI101 Vision Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Next Generation AI Object Detection powered by YOLOv8</div>', unsafe_allow_html=True)

# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------- SIDEBAR ----------
st.sidebar.title("Control Panel")

confidence = st.sidebar.slider(
    "Confidence",
    0.1,1.0,0.3
)

st.sidebar.markdown("---")
st.sidebar.write("Model: YOLOv8 Nano")
st.sidebar.success("System Ready")

# ---------- UPLOAD AREA ----------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1,col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Original Image")
        st.image(image,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.spinner("AI is analyzing the image..."):
        start=time.time()
        results=model(img_array,conf=confidence)
        plotted=results[0].plot()
        end=time.time()

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Detection Result")
        st.image(plotted,use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- METRICS ----------
    boxes=results[0].boxes

    col3,col4,col5=st.columns(3)

    col3.metric("Objects Detected",len(boxes))
    col4.metric("Inference Time",f"{round(end-start,2)} s")
    col5.metric("Image Resolution",f"{image.size[0]}x{image.size[1]}")

    st.success("Detection Complete")
