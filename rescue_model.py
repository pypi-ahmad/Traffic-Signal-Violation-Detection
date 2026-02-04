import shutil
import os

# The path found in your logs
wrong_path = r"D:\Workspace\Github\Car-Object-Detection\runs\detect\runs\train\traffic_violation_large\weights\best.pt"
target_path = "best.pt"

if os.path.exists(wrong_path):
    print(f"✅ Found the missing model at: {wrong_path}")
    shutil.copy(wrong_path, target_path)
    print(f"🚀 Successfully moved model to: {os.path.abspath(target_path)}")
else:
    print(f"❌ Could not find file at: {wrong_path}")
    print("Trying to search recursively in current folder...")
    
    # Fallback: Find any 'best.pt' created recently
    for root, dirs, files in os.walk("."):
        if "best.pt" in files:
            found = os.path.join(root, "best.pt")
            print(f"Found candidate: {found}")
            shutil.copy(found, target_path)
            print(f"🚀 Moved {found} to root!")
            break