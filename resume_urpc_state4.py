import argparse
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Inject custom modules to allow parsing YOLOv11m modifications
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

def main():
    print("\n" + "="*50)
    print("RESUMING STATE 4 (CLASS-BALANCED LOSS) TRAINING ON URPC")
    print("="*50 + "\n")
    
    # Wajib nyalakan flag ini agar metode patching Class-Balanced Loss aktif!
    custom_modules.USE_CLASS_BALANCED = True
    
    # Load the latest URPC checkpoint otomatis
    model = YOLO("runs/detect/URPC_state4/weights/last.pt")
    
    print("Resuming Training from last checkpoint...")
    model.train(resume=True)
    
if __name__ == "__main__":
    main()
