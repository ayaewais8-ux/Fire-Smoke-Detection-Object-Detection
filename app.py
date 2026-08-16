"""
Fire & Smoke Detection — Streamlit Demo App (YOLOv11n)
========================================================
Supports:
  - Single image upload
  - Video file upload (frame-by-frame processing)
  - Live webcam feed
Model: YOLOv11n (chosen for its balance of accuracy, size, and speed —
see the project's comparison report for the full analysis).

REPO LAYOUT THIS FILE EXPECTS
------------------------------
fire-detection-project/
├── app/
│   ├── app.py              <- this file
│   ├── requirements.txt
│   └── models/
│       └── yolov11n_best.pt

HOW TO RUN
----------
1. Clone the repo and cd into it.
2. Create/activate a virtual environment.
3. Install dependencies:
       pip install -r app/requirements.txt
4. From the repo root, run:
       streamlit run app/app.py
   (Do NOT run this file with plain `python app.py` — Streamlit apps must be
   launched with the `streamlit run` command.)
"""

import os
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
import torch

# ======================================================================
# 0. CONFIGURATION
# ======================================================================
# Path to the model weights, relative to this file — works on every
# teammate's machine as long as the repo folder structure is kept intact.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(THIS_DIR, "models", "yolov11n_best.pt")

CLASS_COLORS = {"fire": (0, 0, 255), "smoke": (128, 128, 128)}  # BGR for OpenCV
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Fire & Smoke Detection", layout="wide")


# ======================================================================
# 1. MODEL LOADER (cached so it only loads once per session)
# ======================================================================
@st.cache_resource(show_spinner="Loading YOLOv11n...")
def load_model():
    if not os.path.isfile(MODEL_PATH):
        return None
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


# ======================================================================
# 2. INFERENCE + DRAWING
# ======================================================================
def run_inference(model, frame_bgr: np.ndarray, conf_thresh: float):
    detections = []
    results = model.predict(frame_bgr, conf=conf_thresh, verbose=False, device=DEVICE)
    r = results[0]
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        cls_name = r.names.get(cls_id, str(cls_id))
        detections.append((x1, y1, x2, y2, cls_name, score))
    return detections


def draw_detections(frame_bgr: np.ndarray, detections):
    out = frame_bgr.copy()
    for x1, y1, x2, y2, cls_name, score in detections:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = CLASS_COLORS.get(cls_name, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


# ======================================================================
# 3. SIDEBAR — settings
# ======================================================================
st.sidebar.title("Settings")
conf_thresh = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.5, 0.05)
mode = st.sidebar.radio("Input source", ["Image", "Video file", "Live webcam"])

st.sidebar.markdown("---")
st.sidebar.caption(f"Model: **YOLOv11n**")
st.sidebar.caption(f"Device: **{DEVICE}**")

model = load_model()
if model is None:
    st.sidebar.error(
        f"Model weights not found at:\n{MODEL_PATH}\n\n"
        f"Make sure 'yolov11n_best.pt' is inside app/models/ in the repo."
    )

# ======================================================================
# 4. MAIN AREA
# ======================================================================
st.title("🔥 Fire & Smoke Detection — YOLOv11n")
st.caption(
    "YOLOv11n was chosen for deployment based on a full comparison across "
    "8 models (YOLOv8 variants, YOLOv11 variants, Faster R-CNN, SSD MobileNet) "
    "— see the project report for details."
)

if model is None:
    st.stop()

# ---------------------------------------------------------------- IMAGE
if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection..."):
            start = time.time()
            detections = run_inference(model, frame_bgr, conf_thresh)
            elapsed = time.time() - start

        result_bgr = draw_detections(frame_bgr, detections)
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Original")
        with col2:
            st.image(result_rgb, caption="Detection result")

        st.info(f"{len(detections)} object(s) detected in {elapsed*1000:.1f} ms")
        for x1, y1, x2, y2, cls_name, score in detections:
            st.write(f"- **{cls_name}** — confidence {score:.2f}")

# ---------------------------------------------------------------- VIDEO FILE
elif mode == "Video file":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tfile.write(uploaded.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_window = st.image([])
        progress = st.progress(0)
        stop_button = st.button("Stop")

        frame_idx = 0
        while cap.isOpened():
            if stop_button:
                break
            ret, frame_bgr = cap.read()
            if not ret:
                break

            detections = run_inference(model, frame_bgr, conf_thresh)
            result_bgr = draw_detections(frame_bgr, detections)
            frame_window.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

            frame_idx += 1
            if total_frames > 0:
                progress.progress(min(1.0, frame_idx / total_frames))

        cap.release()
        os.unlink(tfile.name)
        st.success("Video processing finished.")

# ---------------------------------------------------------------- LIVE WEBCAM
else:
    st.warning("Live webcam requires a camera connected to the machine running this app.")
    run = st.checkbox("Start webcam")
    frame_window = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access the webcam. Check that no other app is using it.")
        else:
            while run:
                ret, frame_bgr = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                detections = run_inference(model, frame_bgr, conf_thresh)
                result_bgr = draw_detections(frame_bgr, detections)
                frame_window.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))

                # Re-check the checkbox each loop iteration so "Stop" is honored
                run = st.session_state.get("Start webcam", run)
            cap.release()
