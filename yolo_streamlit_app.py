import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import pandas as pd

# Load your trained model (make sure the .pt file exists in the same directory)
model = YOLO("yolov8-weapon-detector.pt")

st.set_page_config(page_title="Real-Time Weapon Detection with YOLOv8", layout="wide")

# Sidebar: Upload & Settings
with st.sidebar:
    st.header("Upload & Settings")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.01)
    iou_threshold = st.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.01)

    st.markdown("---")
    st.caption("Built with Streamlit + YOLOv8")

# Title and Instructions
st.markdown("## Real-Time Weapon Detection with YOLOv8")
st.write("Upload an image to detect weapons using your trained model.")

# Run detection if an image is uploaded
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_img_path = temp_file.name

    col1, col2 = st.columns(2)

    # Show original image
    with col1:
        st.subheader("📷 Original Image")
        st.image(temp_img_path, use_column_width=True)

    # Run YOLOv8 inference
    results = model(temp_img_path, conf=confidence_threshold, iou=iou_threshold)[0]

    # Save prediction result image
    result_img_path = temp_img_path.replace(".jpg", "_pred.jpg")
    results.save(filename=result_img_path)

    # Show prediction image
    with col2:
        st.subheader("🔍 Detection Result")
        st.image(result_img_path, use_column_width=True)

    # Display detection summary
    df = results.to_df()
    if not df.empty:
        st.subheader("🧾 Detection Summary")
        # Optional: Rename columns
        df.rename(columns={"name": "Weapon", "confidence": "Confidence", "box": "Bounding Box"}, inplace=True)
        st.dataframe(df[["Weapon", "Confidence", "Bounding Box"]])
    else:
        st.warning("No weapons detected in the uploaded image.")
