# Python In-built packages
from pathlib import Path
import PIL
import pandas as pd
from collections import Counter

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
        class_name = class_names.get(cls, "Unknown")
        
        if selected_class and class_name != selected_class:
            continue  # Skip boxes that don't match the selected class

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = ' '.join(class_name.split()[:2])

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
st.title("Service Key Detection")



# Additional Settings Dropdown
with st.sidebar.expander("Additional Settings"):
    confidence = float(st.slider(
        "Select Model Confidence (%)", 25, 100, 40)) / 100

st.sidebar.header("Upload Image")

# Image Upload
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

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
    st.session_state['detections'] = None
    st.session_state['cls_names'] = []
    st.session_state['image_np'] = None

# Layout for Original and Detected Images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    try:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path).convert("RGB")
            st.image(default_image, caption="Default Image", use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img).convert("RGB")
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    st.subheader("Detected Image")
    try:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path).convert("RGB")
            st.image(default_detected_image, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=confidence)
                boxes = res[0].boxes
                
                if not boxes:
                    st.warning("No objects detected.")
                    st.session_state['detections'] = None
                    st.session_state['cls_names'] = []
                    st.session_state['image_np'] = None
                else:
                    # Convert PIL Image to OpenCV format (numpy array)
                    image_np = np.array(uploaded_image)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                    
                    # Store detections in session state
                    st.session_state['detections'] = boxes
                    st.session_state['cls_names'] = [model.names[int(box.cls[0])] for box in boxes]
                    st.session_state['image_np'] = image_np.copy()
                    
                    # Draw all bounding boxes initially
                    image_np_drawn = plot_boxes(image_np.copy(), boxes, model.names)
                    
                    # Convert back to RGB for displaying in Streamlit
                    image_np_drawn = cv2.cvtColor(image_np_drawn, cv2.COLOR_BGR2RGB)
                    
                    # Display the processed image
                    st.image(image_np_drawn, caption='Detected Image', use_column_width=True)
                    
                    # Create a table of classes and their quantities
                    cls_counts = Counter(st.session_state['cls_names'])
                    
                    df = pd.DataFrame(cls_counts.items(), columns=['Class', 'Quantity'])
                    total_quantity = df['Quantity'].sum()
                    
                    # Append total row
                    total_row = pd.DataFrame([{'Class': 'Total', 'Quantity': total_quantity}])
                    df = pd.concat([df, total_row], ignore_index=True)
                    
                    st.subheader("Detection Summary")
                    st.table(df)
    except Exception as ex:
        st.error("Error occurred during object detection.")
        st.error(ex)

# Interactive Filtering
if st.session_state['detections']:
    st.subheader("Filter Detections by Class")
    unique_classes = sorted(list(set(st.session_state['cls_names'])))
    selected_class = st.selectbox("Select a class to filter", ["All"] + unique_classes)

    # Retrieve stored image and detections
    image_np_filtered = st.session_state['image_np'].copy()
    boxes = st.session_state['detections']
    class_names = model.names

    if selected_class != "All":
        # Filter boxes for the selected class
        filtered_boxes = [box for box, cls in zip(boxes, st.session_state['cls_names']) if cls == selected_class]
    else:
        # No filtering
        filtered_boxes = boxes

    if filtered_boxes:
        # Draw filtered bounding boxes
        image_np_filtered = plot_boxes(image_np_filtered.copy(), filtered_boxes, class_names, selected_class if selected_class != "All" else None)
        
        # Convert back to RGB for displaying in Streamlit
        image_np_filtered = cv2.cvtColor(image_np_filtered, cv2.COLOR_BGR2RGB)
        
        st.image(image_np_filtered, caption=f'Detected Image - {"All Classes" if selected_class == "All" else selected_class}', use_column_width=True)
    else:
        st.warning(f"No detections found for class: {selected_class}")
