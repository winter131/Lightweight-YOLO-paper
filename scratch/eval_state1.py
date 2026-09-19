import os
import sys
sys.path.insert(0, '.')
from ultralytics import YOLO

def main():
    model_path = r'runs/detect/URPC_state1/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found")
        return
        
    model = YOLO(model_path)
    print("Evaluating State 1 best.pt on URPC dataset...")
    results = model.val(data='dataset/urpc_raw/data.yaml', imgsz=640, device='0', workers=0)
    
    metrics = results.results_dict
    print("\n--- STATE 1 OVERALL METRICS ---")
    print(f"Precision: {metrics.get('metrics/precision(B)', 0)*100:.2f}%")
    print(f"Recall: {metrics.get('metrics/recall(B)', 0)*100:.2f}%")
    print(f"mAP@0.5: {metrics.get('metrics/mAP50(B)', 0)*100:.2f}%")
    print(f"mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0)*100:.2f}%")
    
    print("\n--- PER-CLASS METRICS ---")
    class_names = results.names
    # results.box.maps / per-class metrics
    for i, name in class_names.items():
        p = results.box.p[i] if hasattr(results.box, 'p') and len(results.box.p) > i else 0
        r = results.box.r[i] if hasattr(results.box, 'r') and len(results.box.r) > i else 0
        ap50 = results.box.ap50[i] if hasattr(results.box, 'ap50') and len(results.box.ap50) > i else 0
        ap = results.box.ap[i] if hasattr(results.box, 'ap') and len(results.box.ap) > i else 0
        print(f"Class {i} ({name}): P={p*100:.2f}%, R={r*100:.2f}%, mAP50={ap50*100:.2f}%, mAP50-95={ap*100:.2f}%")

if __name__ == '__main__':
    main()
