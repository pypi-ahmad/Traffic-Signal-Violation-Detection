# 🚦 Traffic Violation Detection System (SOTA YOLOv8)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-SOTA-green?logo=ultralytics&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

## 📌 Overview

The **Traffic Violation Detection System** is a state-of-the-art computer vision application designed to automate traffic enforcement. It leverages a **Hybrid Dual-Model Architecture** to detect both signal violations (running red lights) and behavioral violations (safety gear compliance and dangerous driving).

This system is built for real-time performance, utilizing **NVIDIA GPU acceleration** and a user-friendly **Streamlit** dashboard for control and monitoring.

---

## 🏗️ Core Architecture

This project employs a robust dual-inference strategy to maximize detection capabilities:

1.  **Vehicle Detection Model (`yolov8n.pt`)**:
    *   **Role**: Tracks standard road users (Cars, Motorcycles, Buses, Trucks).
    *   **Function**: Monitors traffic flow and enforces **Red Light Compliance** by tracking vehicle movement across a defined stop line.

2.  **Custom Violation Model (`best.pt`)**:
    *   **Role**: Detects specific behavioral infractions.
    *   **Classes**:
        *   🚫 `No Helmet`
        *   🛵 `Triple Riding`
        *   📱 `Using Mobile`
        *   🏍️ `Wheeling` (Stunts/Dangerous Driving)
    *   **Function**: Flags these violations **immediately**, regardless of the traffic signal state.

---

## ✨ Key Features

*   **Real-Time Inference**: Processes live webcam feeds or uploaded video files with low latency.
*   **Interactive Controls**:
    *   **🚦 Signal Control**: Manually toggle traffic light state (RED/GREEN) to simulate intersection logic.
    *   **📏 Smart Stop Line**: Draggable slider to adjust the stop line position dynamically based on camera angle.
*   **Intelligent Violation Logic**:
    *   **Context-Aware**: Red light violations are only counted when the signal is RED.
    *   **Safety-First**: Severe violations (e.g., No Helmet) are detected continuously.
*   **Visual Feedback**: Real-time bounding boxes, violation labels, and an on-screen counter.

---

## 🚀 Installation

### Prerequisites
*   Python 3.8+
*   NVIDIA GPU (Recommended for high FPS) with CUDA 11.8+
*   [Git LFS](https://git-lfs.com/) (For downloading large model weights)

### 1. Clone the Repository
```bash
git clone https://github.com/pypi-ahmad/Traffic-Signal-Violation-Detection.git
cd Traffic-Signal-Violation-Detection
```

### 2. Environment Setup
It is recommended to use a virtual environment.
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

Run the dashboard application:
```bash
streamlit run app.py
```

### Dashboard Controls
1.  **Input Source**: Choose between "Upload Video" or "Webcam/Live".
2.  **Model Confidence**: Adjust the slider (0.0 - 1.0) to filter weak detections.
3.  **Traffic Light State**:
    *   🔴 **RED**: Enforce stop line compliance.
    *   🟢 **GREEN**: Allow traffic flow (only behavioral violations monitored).
4.  **Stop Line Position**: Move the slider to align the red line with the actual road markings.

---

## 🛠️ Training Pipeline (Reproducibility)

To retrain the custom violation model (`best.pt`), follow these steps:

### 1. Data Ingestion
Download the curated dataset from Roboflow.
```bash
# Requires ROBOFLOW_API_KEY environment variable
python download_data.py
```

### 2. Model Training
Train the **YOLOv8 Large** model. This script is optimized for GPUs (Batch=8, AMP=True).
```bash
python train.py
```
*Artifacts will be saved to `runs/train/traffic_violation_large` and the best model copied to root.*

---

## 🐳 Docker Deployment

Deploy the application in an isolated container.

### 1. Build Image
```bash
docker build -t traffic-violation-system .
```

### 2. Run Container
Map port 8501 for Streamlit and pass GPU access (if available).
```bash
# With GPU Support (NVIDIA Container Toolkit required)
docker run --gpus all -p 8501:8501 traffic-violation-system

# CPU Only
docker run -p 8501:8501 traffic-violation-system
```

---

## 📂 Project Structure

```plaintext
Traffic-Signal-Violation-Detection/
├── app.py                # Main Streamlit Application
├── train.py              # Training Script (YOLOv8l)
├── download_data.py      # Dataset Downloader (Roboflow)
├── requirements.txt      # Project Dependencies
├── Dockerfile            # Container Configuration
├── best.pt               # Custom Trained Violation Model
├── yolov8n.pt            # Pre-trained Vehicle Model
├── README.md             # Project Documentation
└── .gitignore            # Git Ignore Rules
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
