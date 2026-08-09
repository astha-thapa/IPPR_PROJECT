"""
streamlit_app.py
-----------------
Deployable web dashboard for the Real-Time Shape Recognition project.
Covers Week-5 "Application Development" requirements:
    Home Page / Upload Image Page / Processing Dashboard /
    Result Page / Report Page

Run locally:
    pip install -r requirements.txt
    streamlit run streamlit_app.py

Deploy for free (so you have a live link for your viva/report):
    1. Push this project folder to a public GitHub repo.
    2. Go to https://share.streamlit.io  (Streamlit Community Cloud)
    3. "New app" -> pick your repo -> main file = streamlit_app.py -> Deploy.
    That's it - you get a public URL to put in your report/slides.

Live webcam:
    Plain `streamlit` can only grab ONE frame at a time (st.camera_input),
    which is fine as a "near real-time" demo page. For a continuously
    streaming live feed inside the browser, this app also offers a
    streamlit-webrtc powered page (auto-detected; installs separately -
    see requirements.txt). If webrtc isn't installed, that page is
    hidden automatically and you still have the Upload + Camera-snapshot
    pages, which are enough to satisfy the guideline.
"""

import time
import numpy as np
import cv2
import streamlit as st
import joblib

from shape_utils import process_frame, extract_features, get_contours, preprocess

st.set_page_config(page_title="Real-Time Shape Recognition", layout="wide")

# ---------------------------------------------------------------------
# Load trained model once
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/shape_classifier.pkl")
        scaler = joblib.load("models/scaler.pkl")
        le = joblib.load("models/label_encoder.pkl")
        return model, scaler, le
    except FileNotFoundError:
        return None, None, None

model, scaler, le = load_model()

# ---------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------
st.sidebar.title("Shape Recognition System")
pages = ["Home", "Upload Image", "Live Camera (snapshot)", "Result & Metrics", "About / Report"]
try:
    import streamlit_webrtc  # noqa
    pages.insert(3, "Live Camera (streaming)")
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

page = st.sidebar.radio("Navigate", pages)
use_ml_global = st.sidebar.checkbox("Use trained ML classifier (KNN)", value=False,
                                     disabled=(model is None))
if model is None:
    st.sidebar.warning("No trained model found. Run train_classifier.py first. "
                        "Rule-based classifier will be used regardless of this toggle.")

# ---------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------
if page == "Home":
    st.title("Real-Time Shape Recognition System")
    st.markdown("""
    An Image Processing & Pattern Recognition project that detects and
    classifies geometric shapes (**Circle, Square, Rectangle, Triangle,
    Pentagon, Hexagon, Star**) from images or a live camera feed —
    built entirely with classical computer vision and a classifier
    **trained from scratch**, with **no pre-trained ML/DL models**.

    **Pipeline:**  Input -> Preprocessing -> Feature Extraction -> Classification -> Output

    Use the sidebar to:
    - Upload an image
    - Take a live snapshot from your camera
    - View performance metrics
    """)
    col1, col2, col3 = st.columns(3)
    col1.metric("Shape Classes", "7")
    col2.metric("Classifier", "KNN (from scratch)" if model else "Rule-based")
    col3.metric("Features per shape", "12")

# ---------------------------------------------------------------------
# UPLOAD IMAGE PAGE
# ---------------------------------------------------------------------
elif page == "Upload Image":
    st.title("Upload an Image")
    uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Processing..."):
            output, detections, intermediates = process_frame(
                frame, model, scaler, le, use_ml=use_ml_global
            )

        st.subheader("Processing Dashboard")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        c2.image(intermediates["gray"], caption="Grayscale", use_container_width=True)
        c3.image(intermediates["thresh"], caption="Thresholded", use_container_width=True)
        c4.image(intermediates["closed"], caption="Morph. Closed", use_container_width=True)

        st.subheader("Result")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Shapes", use_container_width=True)

        if detections:
            st.write(f"**{len(detections)} shape(s) detected:**")
            st.table([{"Shape": d["label"], "Bounding Box (x,y,w,h)": d["bbox"]} for d in detections])
        else:
            st.info("No shapes detected. Try an image with clear, high-contrast shapes on a plain background.")

# ---------------------------------------------------------------------
# LIVE CAMERA - SNAPSHOT (works everywhere, no extra install needed)
# ---------------------------------------------------------------------
elif page == "Live Camera (snapshot)":
    st.title("Live Camera - Snapshot Mode")
    st.caption("Click below, allow camera access, and capture a frame for instant recognition.")
    cam_image = st.camera_input("Take a picture")
    if cam_image is not None:
        file_bytes = np.asarray(bytearray(cam_image.getvalue()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        output, detections, _ = process_frame(frame, model, scaler, le, use_ml=use_ml_global)
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), caption="Detected Shapes", use_container_width=True)
        if detections:
            st.table([{"Shape": d["label"], "Bounding Box": d["bbox"]} for d in detections])

# ---------------------------------------------------------------------
# LIVE CAMERA - CONTINUOUS STREAMING (optional, needs streamlit-webrtc)
# ---------------------------------------------------------------------
elif page == "Live Camera (streaming)":
    st.title("Live Camera - Continuous Streaming")
    from streamlit_webrtc import webrtc_streamer
    import av

    class Processor:
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            output, _, _ = process_frame(img, model, scaler, le, use_ml=use_ml_global)
            return av.VideoFrame.from_ndarray(output, format="bgr24")

    webrtc_streamer(key="shape-stream", video_processor_factory=Processor)

# ---------------------------------------------------------------------
# RESULT & METRICS PAGE
# ---------------------------------------------------------------------
elif page == "Result & Metrics":
    st.title("Model Performance Metrics")
    try:
        with open("outputs/accuracy_report.txt") as f:
            st.text(f.read())
        st.image("outputs/confusion_matrix.png", caption="Confusion Matrix")
    except FileNotFoundError:
        st.warning("Run train_classifier.py first to generate metrics and the confusion matrix.")

# ---------------------------------------------------------------------
# ABOUT / REPORT PAGE
# ---------------------------------------------------------------------
elif page == "About / Report":
    st.title("About This Project")
    st.markdown("""
    **Course:** Image Processing and Pattern Recognition (IPPR)
    **Topic:** Real-Time Shape Recognition
    **Constraint:** No pre-trained ML/DL models used.

    **Techniques used:**
    - Grayscale conversion, Gaussian blur, adaptive thresholding, morphological closing
    - Contour detection & polygon approximation (`approxPolyDP`)
    - Geometric features: vertex count, circularity, aspect ratio, extent, solidity
    - Hu Moments (rotation/scale/translation-invariant shape descriptors)
    - Rule-based classifier (explainable baseline)
    - K-Nearest-Neighbours classifier **trained from scratch** on a
      self-generated synthetic dataset (840 images, 7 classes)
    """)
