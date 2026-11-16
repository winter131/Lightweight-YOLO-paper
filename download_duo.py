import os
import yaml
from roboflow import Roboflow

print("[*] Menghubungkan ke Roboflow API...")
rf = Roboflow(api_key="UQ6SBTR3zKBxpYWiexxW")
project = rf.workspace("duo-dataset").project("duo-dataset")
version = project.version(2)

target_dir = os.path.abspath("dataset/duo")
print(f"[*] Memulai unduhan DUO dataset (format YOLOv8) ke: {target_dir} ...")

dataset = version.download("yolov8", location=target_dir)
print(f"[+] Unduhan selesai di: {dataset.location}")

# Periksa & standarisasi file data.yaml
yaml_file = os.path.join(target_dir, "data.yaml")
if os.path.exists(yaml_file):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    
    # Standarisasi path agar kompatibel sempurna dengan Ultralytics
    data["path"] = "dataset/duo"
    data["train"] = "train/images"
    data["val"] = "valid/images"
    if "test" in data:
        data["test"] = "test/images"
        
    with open(yaml_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"[+] data.yaml berhasil distandarisasi untuk Ultralytics: {data['names']}")
