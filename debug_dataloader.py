import os
import time
from pathlib import Path
from PIL import Image

def analyze_dataset(name, path_str):
    print(f"\n--- Analyzing Dataset: {name} ---")
    img_dir = Path(path_str)
    
    if not img_dir.exists():
        print(f"Path does not exist: {img_dir}")
        return
        
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    
    if not img_files:
        print(f"No images found in {img_dir}")
        return
        
    print(f"Total images found: {len(img_files)}")
    
    # Sample up to 100 images to avoid taking too long
    sample_size = min(100, len(img_files))
    sample_files = img_files[:sample_size]
    
    total_size_bytes = 0
    total_pixels = 0
    start_time = time.time()
    
    for img_path in sample_files:
        # File size
        size_bytes = img_path.stat().st_size
        total_size_bytes += size_bytes
        
        # Image resolution
        with Image.open(img_path) as img:
            w, h = img.size
            total_pixels += (w * h)
            
    end_time = time.time()
    
    avg_size_kb = (total_size_bytes / sample_size) / 1024
    avg_pixels = total_pixels / sample_size
    avg_w = (avg_pixels ** 0.5) * (16/9) ** 0.5 # rough estimation assuming 16:9
    
    read_speed = sample_size / (end_time - start_time)
    
    print(f"Avg File Size: {avg_size_kb:.2f} KB")
    print(f"Avg Resolution (pixels): {avg_pixels:,.0f}")
    print(f"Simulated Read Speed: {read_speed:.2f} images/sec (raw PIL open without resize)")
    
if __name__ == "__main__":
    urpc_path = "C:/Users/IT - 0204/Desktop/jaya/ITS/UPRC_2020_thesis/train/images"
    brackish_path = "C:/Users/IT - 0204/Desktop/jaya/ITS/UPRC_2020_thesis/Brackfish_Split/train/images"
    
    analyze_dataset("URPC", urpc_path)
    analyze_dataset("Brackish", brackish_path)
