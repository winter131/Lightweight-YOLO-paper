import os
import argparse
import pandas as pd
from ultralytics import YOLO
import torch
import time

import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Inject custom modules into ultralytics namespace
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock

yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

# Epoch tracking callback
def on_train_epoch_start(trainer):
    custom_modules.CURRENT_EPOCH = trainer.epoch + 1

def main():
    dataset = 'dataset/brackish_raw/data.yaml'
    state = 8
    dataset_name = "Brackish"
    
    print(f"\n{'='*60}")
    print(f"LAUNCHING PROPOSED MODEL (STATE 8) ON BRACKISH DATASET")
    print(f"Goal: Evaluate Cross-Domain Generalization & Efficiency")
    print(f"{'='*60}\n")
    
    config_file = "configs/yolo11m_state8_brackish.yaml"
    run_name = f"{dataset_name}_state8"
    
    # Enable Proposed Model Flags (Axis A + B + C)
    custom_modules.USE_CLASS_BALANCED = True
    custom_modules.USE_REPULSION_LOSS = True
    custom_modules.CURRENT_EPOCH = 1
    custom_modules.MAX_EPOCHS = 100
    
    model = YOLO(config_file)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    
    # Print verified model info
    info = model.info()
    print(f"Verified Model Params: {info[1]/1e6:.2f}M | GFLOPs: {info[3]:.2f}")
    
    print("Starting Training for Brackish State 8 (100 Epochs)...")
    train_results = model.train(
        data=dataset,
        epochs=100,
        imgsz=640,
        batch=8,           # Strictly identical to State 1 Baseline
        amp=True,
        device='0',
        name=run_name,
        exist_ok=True,
        plots=False,
        save_period=5
    )
    
    print("\nTraining completed. Running validation...")
    val_results = model.val(data=dataset, imgsz=640, device='0')
    map50 = val_results.results_dict["metrics/mAP50(B)"]
    map50_95 = val_results.results_dict["metrics/mAP50-95(B)"]
    
    print(f"\n==================================================")
    print(f"BRACKISH STATE 8 EVALUATION COMPLETED:")
    print(f"mAP@0.5: {map50*100:.2f}% (Baseline State 1: 98.70%)")
    print(f"mAP@0.5:0.95: {map50_95*100:.2f}% (Baseline State 1: 84.20%)")
    print(f"==================================================")

if __name__ == '__main__':
    main()
