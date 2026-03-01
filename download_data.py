import os
import sys
from roboflow import Roboflow

def download_dataset():
    """
    Downloads the Traffic Violation dataset from Roboflow in YOLOv8 format.
    Requires ROBOFLOW_API_KEY to be set in environment variables or provided via input.
    """
    print("Initializing Roboflow data ingestion...")

    # 1. Retrieve API Key
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    
    if not api_key:
        print("ROBOFLOW_API_KEY not found in environment variables.")
        if not sys.stdin.isatty():
            print("Error: Non-interactive shell detected and ROBOFLOW_API_KEY is not set.")
            sys.exit(1)

        api_key = input("Please enter your Roboflow API Key: ").strip()
        
        if not api_key:
            print("Error: API Key is required to download the dataset.")
            sys.exit(1)
    
    try:
        # 2. Initialize Roboflow client
        rf = Roboflow(api_key=api_key)
        
        # 3. Access the specific workspace and project
        # Workspace: traffic-violation-detection
        # Project: tvd-kp9qw
        print("Accessing workspace: 'traffic-violation-detection', project: 'tvd-kp9qw'...")
        project = rf.workspace("traffic-violation-detection").project("tvd-kp9qw")
        
        # 4. Download the dataset
        # Using version 2 as requested. Change version number if a newer one is desired.
        # Format: yolov8
        version_number = 2
        print(f"Downloading dataset version {version_number} in 'yolov8' format...")
        dataset = project.version(version_number).download("yolov8")
        
        print(f"Success! Dataset downloaded to: {dataset.location}")
        
    except Exception as e:
        print(f"An error occurred during dataset download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_dataset()
