import time
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks
import ultralytics.utils.nms as yolo_nms
from cluster_nms import patch_nms, unpatch_nms

# Register custom modules
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

def benchmark_model(weights_path, model_name, is_proposed=False, half=False, imgsz=640, warmup_runs=50, num_runs=200):
    print(f"\n=======================================================")
    print(f"BENCHMARKING: {model_name} (Half/FP16: {half})")
    print(f"Weights: {weights_path}")
    print(f"Batch: 1 | Imgsz: {imgsz}x{imgsz} | Warmup: {warmup_runs} | Runs: {num_runs}")
    print(f"=======================================================")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    yolo_wrapper = YOLO(weights_path)
    model = yolo_wrapper.model.to(device).eval()
    
    if half:
        model = model.half()
        dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16, device=device)
    else:
        model = model.float()
        dummy_input = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float32, device=device)
        
    # Configure NMS
    if is_proposed:
        patch_nms()
        nms_type = "Cluster-NMS"
    else:
        unpatch_nms()
        nms_type = "Standard Greedy-NMS"
        
    # CUDA Events for hardware-level timing
    start_infer = torch.cuda.Event(enable_timing=True)
    end_infer = torch.cuda.Event(enable_timing=True)
    start_nms = torch.cuda.Event(enable_timing=True)
    end_nms = torch.cuda.Event(enable_timing=True)
    
    # 1. Warm-up Phase
    print(f"Running {warmup_runs} warm-up iterations to lock GPU clocks and warm cache...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            preds = model(dummy_input)
            _ = yolo_nms.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
    torch.cuda.synchronize()
    
    # 2. Benchmark Phase
    infer_times = []
    nms_times = []
    e2e_times = []
    
    print(f"Measuring {num_runs} consecutive runs...")
    with torch.no_grad():
        for _ in range(num_runs):
            # Synchronize before starting
            torch.cuda.synchronize()
            
            # Forward Pass (Inference)
            start_infer.record()
            preds = model(dummy_input)
            end_infer.record()
            
            # NMS Post-Processing
            start_nms.record()
            _ = yolo_nms.non_max_suppression(preds, conf_thres=0.25, iou_thres=0.45)
            end_nms.record()
            
            # Wait for all CUDA kernels to finish
            torch.cuda.synchronize()
            
            t_infer = start_infer.elapsed_time(end_infer)  # milliseconds
            t_nms = start_nms.elapsed_time(end_nms)        # milliseconds
            t_e2e = t_infer + t_nms                       # milliseconds
            
            infer_times.append(t_infer)
            nms_times.append(t_nms)
            e2e_times.append(t_e2e)
            
    # Unpatch NMS after run
    unpatch_nms()
    
    # Compute statistics
    infer_mean = np.mean(infer_times)
    infer_std = np.std(infer_times)
    nms_mean = np.mean(nms_times)
    nms_std = np.std(nms_times)
    e2e_mean = np.mean(e2e_times)
    e2e_std = np.std(e2e_times)
    fps = 1000.0 / e2e_mean
    
    print(f"-> Inference (Forward): {infer_mean:.2f} ± {infer_std:.2f} ms")
    print(f"-> Post-Process ({nms_type}): {nms_mean:.2f} ± {nms_std:.2f} ms")
    print(f"-> End-to-End Latency: {e2e_mean:.2f} ± {e2e_std:.2f} ms")
    print(f"-> End-to-End FPS: {fps:.1f} FPS")
    
    return {
        'Model': model_name,
        'Precision': 'FP16' if half else 'FP32',
        'NMS': nms_type,
        'Infer Latency (ms)': f"{infer_mean:.2f} ± {infer_std:.2f}",
        'NMS Latency (ms)': f"{nms_mean:.2f} ± {nms_std:.2f}",
        'End-to-End Latency (ms)': f"{e2e_mean:.2f} ± {e2e_std:.2f}",
        'Throughput (FPS)': f"{fps:.1f}",
        'e2e_mean': e2e_mean,
        'fps_raw': fps
    }

def main():
    benchmarks = [
        # URPC Models
        {
            'path': 'runs/detect/URPC_state1/weights/best.pt',
            'name': 'Baseline YOLOv11m (URPC)',
            'proposed': False
        },
        {
            'path': 'runs/detect/URPC_state8/weights/best.pt',
            'name': 'Light-YOLOv11m State 8 (URPC)',
            'proposed': True
        },
        # Brackish Models
        {
            'path': 'runs/detect/Brackish_state1/weights/best.pt',
            'name': 'Baseline YOLOv11m (Brackish)',
            'proposed': False
        },
        {
            'path': 'runs/detect/Brackish_state8/weights/best.pt',
            'name': 'Light-YOLOv11m State 8 (Brackish)',
            'proposed': True
        }
    ]
    
    results = []
    
    for b in benchmarks:
        # Run FP32
        res_fp32 = benchmark_model(
            weights_path=b['path'],
            model_name=b['name'],
            is_proposed=b['proposed'],
            half=False,
            imgsz=640,
            warmup_runs=50,
            num_runs=200
        )
        results.append(res_fp32)
        
        # Run FP16
        res_fp16 = benchmark_model(
            weights_path=b['path'],
            model_name=b['name'],
            is_proposed=b['proposed'],
            half=True,
            imgsz=640,
            warmup_runs=50,
            num_runs=200
        )
        results.append(res_fp16)
        
    df = pd.DataFrame(results)
    df.to_csv('rigorous_benchmark_results.csv', index=False)
    
    print("\n\n=======================================================")
    print("FINAL RIGOROUS BENCHMARK SUMMARY (NVIDIA RTX 4050 Laptop)")
    print("=======================================================")
    print(df[['Model', 'Precision', 'NMS', 'Infer Latency (ms)', 'NMS Latency (ms)', 'End-to-End Latency (ms)', 'Throughput (FPS)']].to_string(index=False))
    
if __name__ == '__main__':
    main()
