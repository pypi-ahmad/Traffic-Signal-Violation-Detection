import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# Page Config
st.set_page_config(page_title="Traffic Violation AI 🚦", layout="wide")

# 1. Load Models
@st.cache_resource
def load_models():
    # Model A: Standard Vehicles (for Red Light violations)
    # Classes: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
    model_vehicles = YOLO('yolov8n.pt') 
    
    # Model B: Your Custom Violations (Helmet, Mobile, etc.)
    try:
        model_custom = YOLO('best.pt')
    except:
        model_custom = None
    
    return model_vehicles, model_custom

model_vehicles, model_custom = load_models()

# Sidebar Config
st.sidebar.title("🚦 Control Panel")
confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.35)

# Traffic Light Control
st.sidebar.markdown("### 🚦 Traffic Light")
light_state = st.sidebar.radio("Signal:", ["GREEN 🟢", "RED 🔴"], index=1)
is_red = (light_state == "RED 🔴")

# Input Source
st.sidebar.markdown("### 📹 Input")
input_source = st.sidebar.radio("Source:", ["Upload Video", "Webcam / Live"])

st.title("Traffic Violation Detection System 🚨")

# Display Active Classes
if model_custom:
    st.info(f"🛡️ **Active Violation Detectors:** {', '.join(model_custom.names.values())}")
else:
    st.warning("⚠️ 'best.pt' not found! Only Red Light detection (Cars) will work.")

# Video Processing
def process_video(cap, line_pos_percent):
    st_frame = st.empty()
    col1, col2 = st.columns(2)
    with col1: violation_ph = st.empty()
    with col2: status_ph = st.empty()
        
    total_violations = 0
    violated_ids = set() # Track unique IDs to prevent double counting
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y = int(height * line_pos_percent)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        # --- 1. DETECT STANDARD VEHICLES (For Red Light) ---
        # Classes: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        results_veh = model_vehicles.track(frame, persist=True, classes=[2, 3, 5, 7], conf=confidence, verbose=False)[0]
        
        # --- 2. DETECT CUSTOM VIOLATIONS (Helmet, Mobile, etc.) ---
        results_custom = None
        if model_custom:
            results_custom = model_custom.track(frame, persist=True, conf=confidence, verbose=False)[0]

        # Draw Stop Line
        line_color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.line(frame, (0, line_y), (width, line_y), line_color, 3)
        cv2.putText(frame, "STOP LINE", (10, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, line_color, 2)

        # Helper to process boxes
        def draw_detections(results, is_custom_violation):
            nonlocal total_violations
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                clss = results.boxes.cls.cpu().numpy().astype(int)
                names = results.names
                
                for box, track_id, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    label = names[cls]
                    
                    # Create Unique ID (prepend 'v' or 'c')
                    uid = f"{'c' if is_custom_violation else 'v'}_{track_id}"
                    
                    # Violation Logic
                    violation_text = ""
                    box_color = (0, 255, 0) # Green (Default Safe)
                    
                    # LOGIC A: Is it one of the 4 Custom Classes? (ALWAYS A VIOLATION)
                    if is_custom_violation:
                        violation_text = label.upper() # e.g., "NO HELMET", "WHEELING"
                        box_color = (0, 0, 255) # Red
                        
                    # LOGIC B: Red Light Crossing (Only if Light is Red)
                    cy = y2
                    if is_red and cy > line_y:
                        if not violation_text: # If not already a custom violation
                            violation_text = "RED LIGHT"
                            box_color = (0, 0, 255)
                        else:
                            violation_text += " + RED LIGHT"
                    
                    # Update Counter
                    if violation_text and uid not in violated_ids:
                        total_violations += 1
                        violated_ids.add(uid)
                    
                    # Draw UI
                    # Only draw bounding box if it's a violation OR if we want to show tracked cars
                    if violation_text:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                        cv2.putText(frame, f"VIOLATION: {violation_text}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                    elif not is_custom_violation:
                        # Draw normal cars in Green tracking box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label
                    cv2.putText(frame, f"{label}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Draw Layers
        if results_veh: draw_detections(results_veh, is_custom_violation=False)
        if results_custom: draw_detections(results_custom, is_custom_violation=True)

        # Metrics
        violation_ph.metric("🚨 Total Violations", total_violations)
        if is_red:
            status_ph.error("🔴 RED LIGHT ACTIVE")
        else:
            status_ph.success("🟢 GREEN LIGHT")

        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width='stretch')
    cap.release()

# Main Execution
line_pos = st.sidebar.slider("Line Position", 0.1, 0.9, 0.6)

if input_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if uploaded_file and st.button("Start Analysis"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        process_video(cv2.VideoCapture(tfile.name), line_pos)
elif input_source == "Webcam / Live":
    if st.button("Start Live Feed"):
        process_video(cv2.VideoCapture(0), line_pos)