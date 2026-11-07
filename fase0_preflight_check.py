import os
import time
import torch
import numpy as np
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Register custom modules
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

def create_yolov11s_configs():
    # State 1: Standard YOLOv11s
    yaml_state1 = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [128, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C3k2, [1024, True]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]

  - [[16, 19, 22], 1, Detect, [nc]]
"""
    # State 2: YOLOv11s + Slim-Neck (GSConv + VoVGSCSP)
    yaml_state2 = """nc: 4
scales:
  s: [0.50, 0.50, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [128, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C3k2, [1024, True]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3Ghost, [512, False]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3Ghost, [256, False]]

  - [-1, 1, GhostConv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3Ghost, [512, False]]

  - [-1, 1, GhostConv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3Ghost, [1024, True]]

  - [[16, 19, 22], 1, Detect, [nc]]
"""
    os.makedirs('configs', exist_ok=True)
    with open('configs/yolo11s_state1.yaml', 'w') as f:
        f.write(yaml_state1)
    with open('configs/yolo11s_state2.yaml', 'w') as f:
        f.write(yaml_state2)
    print("Configs created: configs/yolo11s_state1.yaml and configs/yolo11s_state2.yaml")

def profile_model_pytorch(cfg_path, model_name, imgsz=640, num_runs=200):
    print(f"\nProfiling PyTorch FP16 (Fused): {model_name}...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Initialize un-trained model with random weights
    model = YOLO(cfg_path)
    model.fuse()
    model.to('cuda').half()
    
    dummy = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16, device='cuda')
    
    # Warmup
    for _ in range(50):
        _ = model.model(dummy)
    torch.cuda.synchronize()
    
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    
    latencies = []
    for _ in range(num_runs):
        t0.record()
        _ = model.model(dummy)
        t1.record()
        torch.cuda.synchronize()
        latencies.append(t0.elapsed_time(t1))
        
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_gb = vram_mb / 1024
    
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat
    
    # Parameters and FLOPs
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    print(f"[{model_name}] Params: {params:.2f}M | Latency: {mean_lat:.2f} ± {std_lat:.2f} ms | FPS: {fps:.1f} | Peak VRAM: {vram_mb:.1f} MB ({vram_gb:.3f} GB)")
    return {
        'model': model_name,
        'params_m': params,
        'latency_ms': mean_lat,
        'latency_std': std_lat,
        'fps': fps,
        'vram_mb': vram_mb,
        'vram_gb': vram_gb,
        'yolo_obj': model
    }

def try_tensorrt_export(model_obj, output_name, imgsz=640):
    print(f"\nAttempting TensorRT / Engine Export for {output_name}...")
    try:
        engine_path = model_obj.export(format='engine', half=True, dynamic=False, batch=1, imgsz=imgsz)
        print(f"TensorRT Engine Export Success: {engine_path}")
        
        # Profile TensorRT engine
        engine_model = YOLO(engine_path)
        dummy = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16, device='cuda')
        
        # Warmup
        for _ in range(50):
            _ = engine_model(dummy)
        torch.cuda.synchronize()
        
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        
        times = []
        for _ in range(200):
            t0.record()
            _ = engine_model(dummy)
            t1.record()
            torch.cuda.synchronize()
            times.append(t0.elapsed_time(t1))
            
        mean_t = np.mean(times)
        std_t = np.std(times)
        fps = 1000.0 / mean_t
        print(f"[{output_name} TensorRT FP16] Latency: {mean_t:.2f} ± {std_t:.2f} ms | FPS: {fps:.1f}")
        return mean_t, fps
    except Exception as e:
        print(f"TensorRT Export error/fallback: {e}")
        return None, None

def main():
    print("=================================================================")
    print("FASE 0: GERBANG LATENSI (PRE-FLIGHT CHECK)")
    print("Hardware: NVIDIA RTX 4050 Laptop GPU | Batch: 1 | Imgsz: 640x640")
    print("Target JRTIP: Latency <= 4.5 ms | VRAM <= 0.8 GB")
    print("=================================================================")
    
    create_yolov11s_configs()
    
    # 1. State 1: Baseline YOLOv11s
    res1 = profile_model_pytorch('configs/yolo11s_state1.yaml', 'Baseline YOLOv11s (State 1)')
    
    # 2. State 2: YOLOv11s + Slim-Neck
    res2 = profile_model_pytorch('configs/yolo11s_state2.yaml', 'YOLOv11s + Slim-Neck (State 2)')
    
    # 3. Test TensorRT export if engine supports it
    trt1_lat, trt1_fps = try_tensorrt_export(res1['yolo_obj'], 'YOLOv11s_State1')
    trt2_lat, trt2_fps = try_tensorrt_export(res2['yolo_obj'], 'YOLOv11s_State2_SlimNeck')
    
    print("\n=================================================================")
    print("EVALUASI GERBANG LATENSI (DECISION MATRIX)")
    print("=================================================================")
    
    param_reduction = ((res1['params_m'] - res2['params_m']) / res1['params_m']) * 100
    
    print(f"Baseline Params : {res1['params_m']:.2f}M")
    print(f"Slim-Neck Params: {res2['params_m']:.2f}M (Turun {param_reduction:.1f}%)")
    print(f"PyTorch FP16 Latency Baseline : {res1['latency_ms']:.2f} ms ({res1['fps']:.1f} FPS) | VRAM: {res1['vram_gb']:.3f} GB")
    print(f"PyTorch FP16 Latency Slim-Neck: {res2['latency_ms']:.2f} ms ({res2['fps']:.1f} FPS) | VRAM: {res2['vram_gb']:.3f} GB")
    
    active_lat = trt2_lat if trt2_lat is not None else res2['latency_ms']
    active_engine = "TensorRT FP16" if trt2_lat is not None else "PyTorch FP16 (Fused)"
    
    pass_lat = active_lat <= 4.5
    pass_vram = res2['vram_gb'] <= 0.8
    
    print(f"\nEvaluasi pada Engine: {active_engine}")
    print(f"Target Latensi <= 4.5 ms : {'LOLOS (PASS) [v]' if pass_lat else 'BELUM TEMBUS [x]'} ({active_lat:.2f} ms)")
    print(f"Target VRAM   <= 0.8 GB : {'LOLOS (PASS) [v]' if pass_vram else 'BELUM TEMBUS [x]'} ({res2['vram_gb']:.3f} GB)")
    
    if pass_lat and pass_vram:
        print("\n--> KESIMPULAN: GERBANG TERBUKA (EMARK TERBUKTI). LAYAK LANJUT TRAINING!")
    else:
        print(f"\n--> KESIMPULAN: Selisih latensi terdeteksi. Silakan review hasil.")

if __name__ == '__main__':
    main()

