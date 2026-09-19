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
    dataset = 'dataset/urpc_raw/data.yaml'
    state = 3
    dataset_name = "URPC"
    
    print(f"\n{'='*60}")
    print(f"LAUNCHING WINNING STATE 3 (Scheduled IoG-Gated Repulsion Loss)")
    print(f"Goal: Beat Baseline 84.64% mAP@0.5")
    print(f"{'='*60}\n")
    
    config_file = f"configs/yolo11m_state{state}.yaml"
    run_name = f"{dataset_name}_state3_winning"
    
    custom_modules.USE_CLASS_BALANCED = False
    custom_modules.USE_REPULSION_LOSS = True
    custom_modules.CURRENT_EPOCH = 1
    custom_modules.MAX_EPOCHS = 100
    
    model = YOLO(config_file)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    
    print("Starting Training for State 3 Winning (100 Epochs)...")
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
        save_period=5
    )
    
    print("\nTraining completed. Running validation...")
    val_results = model.val(data=dataset, imgsz=640, device='0')
    map50 = val_results.results_dict["metrics/mAP50(B)"]
    map50_95 = val_results.results_dict["metrics/mAP50-95(B)"]
    
    print(f"\n==================================================")
    print(f"STATE 3 WINNING RESULTS:")
    print(f"mAP@0.5: {map50*100:.2f}% (Baseline: 84.64%)")
    print(f"mAP@0.5:0.95: {map50_95*100:.2f}% (Baseline: 50.33%)")
    print(f"==================================================")

if __name__ == '__main__':
    main()
