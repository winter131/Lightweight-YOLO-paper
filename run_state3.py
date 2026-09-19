import os
import argparse
import pandas as pd
from ultralytics import YOLO
import torch
import time

# Import custom NMS monkey patches
from cluster_nms import patch_nms, unpatch_nms
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Inject custom modules
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock

yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

def main():
    dataset = 'dataset/urpc_raw/data.yaml'
    state = 3
    dataset_name = "URPC"
    
    print(f"\n{'='*50}")
    print(f"RUNNING STATE {state} (Baseline + Repulsion Loss) ON DATASET: {dataset_name}")
    print(f"{'='*50}\n")
    
    config_file = f"configs/yolo11m_state{state}.yaml"
    run_name = f"{dataset_name}_state{state}"
    
    # --- APPLY ABLATION FLAGS FOR STATE 3 ---
    custom_modules.USE_CLASS_BALANCED = False
    custom_modules.USE_REPULSION_LOSS = True
    
    uses_cluster_nms = True
    
    model = YOLO(config_file)
    
    print("Starting Training for State 3 (100 Epochs)...")
    train_results = model.train(
        data=dataset,
        epochs=100,
        imgsz=640,
        batch=8,
        amp=True,
        device='0',
        name=run_name,
        exist_ok=True,
        plots=False,
        save_period=10
    )
    
    print("\nStarting Evaluation for State 3...")
    patch_nms()
    eval_results = model.val(data=dataset, imgsz=640, device='0')
    map50 = eval_results.results_dict["metrics/mAP50(B)"]
    map50_95 = eval_results.results_dict["metrics/mAP50-95(B)"]
    
    print(f"\nSTATE 3 EVALUATION COMPLETED:")
    print(f"mAP@0.5: {map50*100:.2f}%")
    print(f"mAP@0.5:0.95: {map50_95*100:.2f}%")

if __name__ == '__main__':
    main()
