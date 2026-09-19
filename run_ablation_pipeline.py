import os
import argparse
import pandas as pd
from ultralytics import YOLO
import torch
import time

# Import custom NMS monkey patches
from cluster_nms import patch_nms, unpatch_nms
# To ensure custom modules are recognized, ultralytics parses them dynamically 
# if they are in the globals or if we override the parse_model function.
# An easier way in Ultralytics without changing source is to inject the custom classes into ultralytics nn modules.
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Inject custom modules so YOLO parses them from the YAML
# We use GhostConv and C3Ghost as the names in the YAML because parse_model 
# checks a hardcoded list of supported classes. Since GhostConv and C3Ghost are in that list, 
# replacing them in the tasks module makes parse_model correctly extract their channels.
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock

# Must also inject into tasks globals for eval() in parse_model
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

def main(args):
    datasets = ['dataset/urpc_raw/data.yaml']
    states = [8]
    
    results_list = []
    
    # Iterate over datasets
    for dataset in datasets:
        dataset_name = "URPC" if "brackish" not in dataset else "Brackish"
        print(f"\n{'='*50}")
        print(f"STARTING EVALUATION ON DATASET: {dataset_name}")
        print(f"{'='*50}\n")
        
        # Iterate over the states
        for state in states:
            print(f"\n--- Running State {state} on {dataset_name} ---")
            config_file = f"configs/yolo11m_state{state}.yaml"
            run_name = f"{dataset_name}_state{state}"
            
            # --- APPLY 3-AXIS ABLATION FLAGS ---
            custom_modules.USE_CLASS_BALANCED = state in [4, 6, 7, 8]
            custom_modules.USE_REPULSION_LOSS = state in [3, 5, 7, 8]
            
            # Cluster-NMS is standard in the new plan
            uses_cluster_nms = True
            
            # Initialize Model
            model = YOLO(config_file)
            
            # Extract parameters and GFLOPs before training
            model_info = model.info()
            # Ultralytics model.info() returns a tuple: (layers, parameters, gradients, GFLOPs)
            params_count = model_info[1] / 1e6  # in Millions
            gflops = model_info[3]
            
            epochs = 1 if args.debug else args.epochs
            
            # 1. Train the model
            print("Starting Training...")
            train_results = model.train(
                data=dataset,
                epochs=epochs,
                imgsz=640,
                batch=args.batch,      # strict control: identical batch size
                amp=True,              # identical mixed precision
                device='0',
                name=run_name,
                exist_ok=True,
                plots=False,
                save_period=10         # Save checkpoint every 10 epochs
            )
            
            # 2. Evaluate the model
            print("Starting Evaluation...")
            # Patch NMS if required for this state
            if uses_cluster_nms:
                print("Applying Cluster-NMS Patch for evaluation...")
                patch_nms()
            
            # Force validation to measure latency properly
            # Proper methodology: warm-up to compile CUDA kernels, then averaged synchronized runs.
            dummy_input = torch.zeros((1, 3, 640, 640)).to('cuda')
            
            # 1. Warm-up
            for _ in range(10):
                _ = model(dummy_input)
                
            # 2. Measured phase
            torch.cuda.synchronize()
            start_t = time.time()
            for _ in range(50):
                _ = model(dummy_input)
            torch.cuda.synchronize()
            
            latency_ms = ((time.time() - start_t) * 1000) / 50
            
            val_results = model.val(data=dataset, imgsz=640, batch=1, device='0')
            
            # Unpatch NMS after evaluation
            if uses_cluster_nms:
                unpatch_nms()
                
            # Extract mAP metrics
            map50 = val_results.results_dict['metrics/mAP50(B)'] * 100
            map5095 = val_results.results_dict['metrics/mAP50-95(B)'] * 100
            
            # Log results
            res_dict = {
                'Dataset': dataset_name,
                'State': state,
                'mAP@0.5 (%)': map50,
                'mAP@0.5:0.95 (%)': map5095,
                'Parameters (M)': params_count,
                'GFLOPs': gflops,
                'Latency (ms)': latency_ms
            }
            results_list.append(res_dict)
            print(f"State {state} Results: {res_dict}")
            
    # Save all results to CSV
    df = pd.DataFrame(results_list)
    df.to_csv("ablation_results.csv", index=False)
    print("\\nAll runs completed successfully. Results saved to ablation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Study Pipeline")
    parser.add_argument('--debug', action='store_true', help="Run in debug mode (1 epoch only)")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=8, help="Batch size (use 8 or 16 for RTX 4050)")
    args = parser.parse_args()
    main(args)
