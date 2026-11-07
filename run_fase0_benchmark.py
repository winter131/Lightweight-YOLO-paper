import os
import time
import torch
import numpy as np
import onnxruntime as ort
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

def export_and_benchmark(cfg_path, model_name, imgsz=640, num_runs=200):
    print(f"\n=======================================================")
    print(f"PROCESSING: {model_name}")
    print(f"=======================================================")
    
    # 1. Initialize PyTorch model (untrained / random weights)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(cfg_path)
    model.to('cuda')
    
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    
    # 2. PyTorch FP16 Profiling
    model_fused = YOLO(cfg_path)
    model_fused.fuse()
    model_fused.to('cuda').half()
    
    dummy_torch = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16, device='cuda')
    for _ in range(50):
        _ = model_fused.model(dummy_torch)
    torch.cuda.synchronize()
    
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    pt_latencies = []
    for _ in range(num_runs):
        t0.record()
        _ = model_fused.model(dummy_torch)
        t1.record()
        torch.cuda.synchronize()
        pt_latencies.append(t0.elapsed_time(t1))
        
    pt_mean = np.mean(pt_latencies)
    pt_std = np.std(pt_latencies)
    pt_fps = 1000.0 / pt_mean
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_gb = vram_mb / 1024
    
    print(f"-> Params: {params_m:.2f}M")
    print(f"-> PyTorch FP16 Latency: {pt_mean:.2f} ± {pt_std:.2f} ms ({pt_fps:.1f} FPS)")
    print(f"-> Peak VRAM: {vram_mb:.1f} MB ({vram_gb:.3f} GB)")
    
    # 3. Export to ONNX
    print("-> Exporting to ONNX format...")
    onnx_path = model.export(format='onnx', imgsz=imgsz, dynamic=False, simplify=True, device=0)
    print(f"-> Exported to: {onnx_path}")
    
    # 4. Benchmark with ONNX Runtime (CUDA Execution Provider)
    print("-> Benchmarking via ONNX Runtime (CUDA Execution Provider)...")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    dummy_np = np.random.randn(1, 3, imgsz, imgsz).astype(np.float32)
    
    # Warmup
    for _ in range(50):
        _ = session.run(None, {input_name: dummy_np})
        
    ort_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_np})
        end = time.perf_counter()
        ort_times.append((end - start) * 1000.0)
        
    ort_mean = np.mean(ort_times)
    ort_std = np.std(ort_times)
    ort_fps = 1000.0 / ort_mean
    
    print(f"-> ONNX Runtime (CUDA) Latency: {ort_mean:.2f} ± {ort_std:.2f} ms ({ort_fps:.1f} FPS)")
    
    return {
        'model': model_name,
        'params_m': params_m,
        'pt_mean': pt_mean,
        'pt_std': pt_std,
        'pt_fps': pt_fps,
        'vram_gb': vram_gb,
        'ort_mean': ort_mean,
        'ort_std': ort_std,
        'ort_fps': ort_fps,
        'onnx_path': onnx_path
    }

def main():
    print("=================================================================")
    print("FASE 0: GERBANG LATENSI (PRE-FLIGHT BENCHMARK)")
    print("Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (CUDA 11.8)")
    print("Batch: 1 | Imgsz: 640x640 | Iterasi: 200 Runs")
    print("Target JRTIP: Latency <= 4.5 ms | VRAM <= 0.8 GB")
    print("=================================================================")
    
    # Run State 1 (Baseline YOLOv11s)
    res1 = export_and_benchmark('configs/yolo11s_state1.yaml', 'State 1: Baseline YOLOv11s')
    
    # Run State 2 (YOLOv11s + Slim-Neck)
    res2 = export_and_benchmark('configs/yolo11s_state2.yaml', 'State 2: YOLOv11s + Slim-Neck')
    
    print("\n=================================================================")
    print("TABEL HASIL LENGKAP FASE 0 (GERBANG LATENSI)")
    print("=================================================================")
    
    param_diff = ((res1['params_m'] - res2['params_m']) / res1['params_m']) * 100
    pt_speedup = ((res1['pt_mean'] - res2['pt_mean']) / res1['pt_mean']) * 100
    ort_speedup = ((res1['ort_mean'] - res2['ort_mean']) / res1['ort_mean']) * 100
    
    print(f"\n1. Parameter Kompresi:")
    print(f"   - Baseline YOLOv11s : {res1['params_m']:.2f} M")
    print(f"   - Slim-Neck YOLOv11s: {res2['params_m']:.2f} M (Turun {param_diff:.1f}%)")
    
    print(f"\n2. VRAM Memory Footprint:")
    print(f"   - Baseline YOLOv11s : {res1['vram_gb']:.3f} GB")
    print(f"   - Slim-Neck YOLOv11s: {res2['vram_gb']:.3f} GB")
    print(f"   - Target JRTIP <= 0.8 GB: {'LOLOS (PASS) [v]' if res2['vram_gb'] <= 0.8 else 'GAGAL [x]'}")
    
    print(f"\n3. Latensi PyTorch FP16 (Fused):")
    print(f"   - Baseline YOLOv11s : {res1['pt_mean']:.2f} ± {res1['pt_std']:.2f} ms ({res1['pt_fps']:.1f} FPS)")
    print(f"   - Slim-Neck YOLOv11s: {res2['pt_mean']:.2f} ± {res2['pt_std']:.2f} ms ({res2['pt_fps']:.1f} FPS) (Speedup: {pt_speedup:+.1f}%)")
    
    print(f"\n4. Latensi ONNX Runtime (CUDA Accelerated):")
    print(f"   - Baseline YOLOv11s : {res1['ort_mean']:.2f} ± {res1['ort_std']:.2f} ms ({res1['ort_fps']:.1f} FPS)")
    print(f"   - Slim-Neck YOLOv11s: {res2['ort_mean']:.2f} ± {res2['ort_std']:.2f} ms ({res2['ort_fps']:.1f} FPS) (Speedup: {ort_speedup:+.1f}%)")
    print(f"   - Target JRTIP <= 4.5 ms: {'LOLOS (PASS) [v]' if res2['ort_mean'] <= 4.5 else 'BELUM TEMBUS [x]'}")
    
    print("\n=================================================================")
    if res2['ort_mean'] <= 4.5 and res2['vram_gb'] <= 0.8:
        print("STATUS: GERBANG TERBUKA PENUH! ARSITEKTUR MEMENUHI SYARAT JRTIP.")
    else:
        print(f"STATUS EVALUASI: Latensi Slim-Neck tercatat {res2['ort_mean']:.2f} ms (Target 4.5 ms).")
    print("=================================================================")

if __name__ == '__main__':
    main()

