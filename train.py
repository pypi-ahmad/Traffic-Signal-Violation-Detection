import os
import shutil
from ultralytics import YOLO

def train_model():
    """
    Trains the YOLOv8l model on the Traffic Violation dataset.
    Optimized for NVIDIA RTX 4060 Laptop GPU (8GB VRAM).
    """
    print("Initializing training process...")

    # 1. Model Selection
    # Load the YOLOv8 Large model. 
    # If the file does not exist locally, it will be downloaded from the Ultralytics release assets.
    model_name = "yolov8l.pt"
    print(f"Loading model: {model_name}")
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        # Fallback suggestion or handling could go here
        raise

    # 2. Dataset Path
    # The dataset was downloaded to 'TVD-2' directory in Phase 1.
    dataset_yaml = "TVD-2/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: Dataset configuration file not found at {dataset_yaml}")
        return

    # 3. Training Configuration
    # Targeted for RTX 4060 (8GB VRAM)
    print("Starting training with the following configuration:")
    print(f"  - Device: 0 (GPU)")
    print(f"  - Batch Size: 8")
    print(f"  - Epochs: 50")
    print(f"  - Image Size: 640")
    print(f"  - AMP (Automatic Mixed Precision): Enabled")

    try:
        results = model.train(
            data=dataset_yaml,
            epochs=50,
            imgsz=640,
            device=0,               # Force usage of GPU 0
            batch=8,                # Batch size 8 fits comfortably in 8GB VRAM for Large models
            amp=True,               # Enable Automatic Mixed Precision for faster training/less memory
            project=os.path.join(os.getcwd(), 'runs', 'train'),   # Root directory for training runs
            name='traffic_violation_large', # subdirectory name
            exist_ok=True,          # Overwrite existing experiment if name exists (optional, keeping it clean)
            plots=True,             # Generate plots (confusion matrix, labels, etc.)
            save=True               # Save checkpoints
        )
        print("Training completed successfully.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        return

    # 4. Save Best Model to Root
    # Ultralytics saves weights in project/name/weights/
    best_weights_path = os.path.join('runs', 'train', 'traffic_violation_large', 'weights', 'best.pt')
    destination_path = 'best.pt'

    if os.path.exists(best_weights_path):
        print(f"Copying best model from {best_weights_path} to {destination_path}...")
        shutil.copy(best_weights_path, destination_path)
        print(f"Best model saved as '{destination_path}' in the project root.")
    else:
        print(f"Warning: Could not find best model at {best_weights_path}")

if __name__ == "__main__":
    train_model()
