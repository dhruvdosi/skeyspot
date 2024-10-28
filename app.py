# Python In-built packages
from pathlib import Path
import PIL
import pandas as pd
from collections import Counter
import zipfile
import os
import logging

# External packages
import streamlit as st
import cv2
import numpy as np

# Local Modules
import settings
import helper

def plot_boxes(image, boxes, class_names, selected_class=None):
    """
    Draws bounding boxes on the image. If selected_class is provided,
    only draws boxes of that class.
    """
    # Define a color map for 34 classes with muted colors
    color_map = {
        0: (255, 220, 180),
        1: (180, 255, 180),
        2: (255, 180, 180),
        3: (180, 255, 255),
        4: (255, 180, 255),
        5: (220, 220, 220),
        6: (255, 240, 180),
        7: (180, 255, 220),
        8: (220, 180, 255),
        9: (180, 220, 255),
        10: (200, 180, 255),
        11: (255, 180, 200),
        12: (180, 255, 200),
        13: (255, 255, 180),
        14: (255, 180, 240),
        15: (180, 255, 255),
        16: (255, 220, 180),
        17: (180, 255, 240),
        18: (200, 255, 180),
        19: (200, 200, 255),
        20: (180, 200, 255),
        21: (255, 180, 240),
        22: (255, 200, 220),
        23: (220, 255, 180),
        24: (240, 255, 180),
        25: (180, 220, 240),
        26: (255, 220, 240),
        27: (240, 220, 180),
        28: (240, 180, 220),
        29: (180, 240, 255),
        30: (180, 220, 220),
        31: (255, 180, 180),
        32: (240, 200, 255),
        33: (200, 200, 220)
    }
    for box in boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = str(cls)  # Display class number instead of name

        color = color_map.get(cls, (255, 255, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

# Cache the model to prevent reloading on every interaction
@st.cache_resource
def load_model_cached(model_path):
    return helper.load_model(model_path)

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.markdown("<h1 style='text-align: center;'>Service Key Detection</h1>", unsafe_allow_html=True)

# Centered container for settings and upload
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Additional Settings
        confidence = float(st.slider(
            "Select Model Confidence (%)", 25, 100, 40, key="confidence_slider")) / 100

        # Image Upload for Multiple Images
        uploaded_images = st.file_uploader(
            "Choose images...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), accept_multiple_files=True, key="image_uploader")

# Set model path to Detection model only
model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = load_model_cached(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()  # Stop execution if model loading fails

# Initialize session state for detections
if 'detections' not in st.session_state:
    st.session_state['detections'] = []
    st.session_state['detection_summary'] = pd.DataFrame()

# Create a temporary directory to store annotated images and CSV files
output_dir = "annotated_outputs"
os.makedirs(output_dir, exist_ok=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('Detect Objects', key="detect_button") and uploaded_images:

        for uploaded_image in uploaded_images:
            try:
                # Open and convert image to OpenCV format
                image = PIL.Image.open(uploaded_image).convert("RGB")
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Perform object detection
                res = model.predict(image, conf=confidence)
                boxes = res[0].boxes

                # Store detections if any boxes are found
                if boxes:
                    st.session_state['detections'] = boxes

                    # Draw bounding boxes on the image
                    image_np_drawn = plot_boxes(image_np.copy(), boxes, model.names)

                    # Convert back to RGB for saving
                    image_np_drawn = cv2.cvtColor(image_np_drawn, cv2.COLOR_BGR2RGB)
                    
                    # Save the annotated image
                    output_image_path = os.path.join(output_dir, f"{uploaded_image.name.split('.')[0]}_annotated.png")
                    PIL.Image.fromarray(image_np_drawn).save(output_image_path)

                    # Create a table of classes, numbers, and their quantities
                    cls_counts = Counter([int(box.cls[0]) for box in boxes])
                    cls_data = [
                        {"Class Label": model.names[cls], "Class Number": cls, "Quantity": count}
                        for cls, count in cls_counts.items()
                    ]
                    total_quantity = sum([item['Quantity'] for item in cls_data])
                    cls_data.append({"Class Label": "Total", "Class Number": "", "Quantity": total_quantity})

                    # Convert to DataFrame
                    df = pd.DataFrame(cls_data)

                    # Save detection summary as a CSV file
                    csv_path = os.path.join(output_dir, f"{uploaded_image.name.split('.')[0]}_detection_summary.csv")
                    df.to_csv(csv_path, index=False)

                    # Store the detection summary using pd.concat for further calculations
                    st.session_state['detection_summary'] = pd.concat([st.session_state['detection_summary'], df], ignore_index=True)

            except Exception as ex:
                logging.error(f"Error processing image {uploaded_image.name}: {ex}")

        # Create a ZIP file of all annotated images and CSV files
        zip_filename = "annotated_images_and_summaries.zip"
        zip_path = os.path.join(output_dir, zip_filename)
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(".png") or file.endswith(".csv"):
                        zipf.write(os.path.join(root, file), arcname=file)

# Dropdown to select the label for calculation
# Centered dropdown for choosing a label
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if not st.session_state['detection_summary'].empty:
        selected_label = st.selectbox(
            "Select a class label to calculate price",
            st.session_state['detection_summary']['Class Label'].unique()
        )

        # Display the total count for the selected class
        selected_class_count = st.session_state['detection_summary'][
            st.session_state['detection_summary']['Class Label'] == selected_label
        ]['Quantity'].sum()
        
        st.write(f"Total count for {selected_label}: {selected_class_count}")

        # Centered unit price input
        unit_price = st.number_input("Enter unit price:", min_value=0.0, value=0.0)

        # Centered "Calculate Total Price" button and total price output
        if st.button("Calculate Total Price", key="calculate_button"):
            total_price = selected_class_count * unit_price
            st.write(f"Total price for {selected_label}: {total_price}")

# Centered download button
col4, col5, col6 = st.columns([1, 2, 1])
with col5:
    if 'zip_path' in locals() and os.path.exists(zip_path):
        with open(zip_path, "rb") as f:
            st.download_button(
                label="Download Annotated Images and CSVs",
                data=f,
                file_name=zip_filename,
                mime="application/zip"
            )

# Clean up temporary files after download
if os.path.exists(output_dir):
    for root, dirs, files in os.walk(output_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
