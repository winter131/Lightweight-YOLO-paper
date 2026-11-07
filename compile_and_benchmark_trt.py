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

def compile_engine(cfg_path, model_name, imgsz=640):
    print(f"\n=======================================================")
    print(f"[EXPORT] Kompilasi {model_name} ke TensorRT FP16...")
    print(f"=======================================================")
    
    # Initialize model with random weights
    model = YOLO(cfg_path)
    model.to('cuda')
    
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"-> Parameter: {params:.2f} M")
    
    # Export directly to TensorRT engine
    t_start = time.time()
    engine_path = model.export(
        format='engine',
        half=True,
        dynamic=False,
        batch=1,
        imgsz=imgsz,
        simplify=True,
        device=0
    )
    t_build = time.time() - t_start
    print(f"-> TensorRT Engine berhasil dibangun dalam {t_build:.1f} detik: {engine_path}")
    return engine_path, params, t_build

def benchmark_trt_engine(engine_path, model_name, imgsz=640, warmup_runs=50, num_runs=200):
    print(f"\n[BENCHMARK] Mengukur Latensi {model_name}...")
    engine_model = YOLO(engine_path)
    
    dummy = torch.randn(1, 3, imgsz, imgsz, dtype=torch.float16, device='cuda')
    
    # Warmup
    print(f"-> Menjalankan {warmup_runs} warm-up runs...")
    for _ in range(warmup_runs):
        _ = engine_model(dummy)
    torch.cuda.synchronize()
    
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    
    latencies = []
    print(f"-> Menjalankan {num_runs} measured runs berturut-turut...")
    for _ in range(num_runs):
        t0.record()
        _ = engine_model(dummy)
        t1.record()
        torch.cuda.synchronize()
        latencies.append(t0.elapsed_time(t1))
        
    mean_lat = np.mean(latencies)
    std_lat = np.std(latencies)
    fps = 1000.0 / mean_lat
    
    # VRAM
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    vram_gb = vram_mb / 1024
    
    print(f"-> Hasil {model_name}: {mean_lat:.2f} ± {std_lat:.2f} ms ({fps:.1f} FPS) | VRAM: {vram_mb:.1f} MB")
    
    return {
        'model': model_name,
        'engine_path': engine_path,
        'latencies': latencies,
        'mean_lat': mean_lat,
        'std_lat': std_lat,
        'fps': fps,
        'vram_mb': vram_mb,
        'vram_gb': vram_gb
    }

def main():
    imgsz = 640
    print("=================================================================")
    print("LANGKAH 1: KOMPILASI & ADU LATENSI TENSORRT FP16 (FASE 0)")
    print("Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (TGP 75W, CUDA)")
    print(f"Resolusi: {imgsz}x{imgsz} | Batch: 1 | Warmup: 50 | Measured: 200 Runs")
    print("Target JRTIP: Latensi <= 4.5 ms | VRAM <= 0.8 GB")
    print("=================================================================")
    
    # 1. Kompilasi State 1 (Baseline YOLOv11s)
    eng1_path, p1, build1 = compile_engine('configs/yolo11s_state1.yaml', 'State 1 (Baseline YOLOv11s)', imgsz=imgsz)
    
    # 2. Kompilasi State 2 (YOLOv11s + Slim-Neck)
    eng2_path, p2, build2 = compile_engine('configs/yolo11s_state2.yaml', 'State 2 (YOLOv11s + Slim-Neck)', imgsz=imgsz)
    
    # 3. Adu Latensi
    res1 = benchmark_trt_engine(eng1_path, 'State 1 (Baseline YOLOv11s)', imgsz=imgsz)
    res2 = benchmark_trt_engine(eng2_path, 'State 2 (YOLOv11s + Slim-Neck)', imgsz=imgsz)
    
    # 4. Analisis Komparasi
    param_diff = ((p1 - p2) / p1) * 100
    lat_diff = ((res1['mean_lat'] - res2['mean_lat']) / res1['mean_lat']) * 100
    fps_diff = ((res2['fps'] - res1['fps']) / res1['fps']) * 100
    
    # Statistical t-test
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(res1['latencies'], res2['latencies'])
    
    pass_lat = res2['mean_lat'] <= 4.5
    pass_vram = res2['vram_gb'] <= 0.8
    
    # Generate Report Markdown
    report_content = f"""# Laporan Fase 0: Hasil Kompilasi & Adu Latensi TensorRT FP16
**Target Jurnal:** Journal of Real-Time Image Processing (JRTIP Springer - Q1)  
**Hardware Uji:** NVIDIA GeForce RTX 4050 Laptop GPU (6GB GDDR6, CUDA Core Ada Lovelace)  
**Konfigurasi Pengujian:** Batch Size = 1, Resolusi $640 \\times 640$, 50 Warmup Runs, 200 Measured Runs.

---

## 1. Tabel Komparasi Hasil TensorRT FP16

| Model Arsitektur | Parameters (M) | Build Time (s) | Latency TensorRT FP16 (ms) | Throughput (FPS) | Peak VRAM Footprint | Status Target JRTIP |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **State 1: Baseline YOLOv11s** | {p1:.2f} M | {build1:.1f} s | {res1['mean_lat']:.2f} ± {res1['std_lat']:.2f} ms | {res1['fps']:.1f} FPS | {res1['vram_mb']:.1f} MB ({res1['vram_gb']:.3f} GB) | Baseline Acuan |
| **State 2: YOLOv11s + Slim-Neck** | **{p2:.2f} M (-{param_diff:.1f}%)** | {build2:.1f} s | **{res2['mean_lat']:.2f} ± {res2['std_lat']:.2f} ms** | **{res2['fps']:.1f} FPS ({fps_diff:+.1f}%)** | **{res2['vram_mb']:.1f} MB ({res2['vram_gb']:.3f} GB)** | {'[PASS]' if pass_lat and pass_vram else '[EVALUASI]'} |

---

## 2. Uji Signifikansi Statistik & Analisis

* **Reduksi Parameter:** Terpangkas sebesar **{param_diff:.1f}%** ({p1:.2f}M -> {p2:.2f}M).
* **Selisih Latensi TensorRT:** Slim-Neck {'lebih cepat' if lat_diff > 0 else 'lebih lambat'} **{abs(lat_diff):.2f}%** ({res1['mean_lat']:.2f} ms vs {res2['mean_lat']:.2f} ms).
* **Uji Hipotesis Statistik (Two-Sample t-Test):**
  * $t$-statistic = `{t_stat:.4f}`
  * $p$-value = `{p_val:.4e}` ({'Signifikan secara statistik (p < 0.05)' if p_val < 0.05 else 'Tidak signifikan (p >= 0.05)'})
* **Evaluasi Target JRTIP:**
  * Target Latensi $\\le 4.5\\text{{ ms}}$: **{'LOLOS (PASS)' if pass_lat else 'BELUM TEMBUS'}** ({res2['mean_lat']:.2f} ms)
  * Target VRAM $\\le 0.8\\text{{ GB}}$: **{'LOLOS TELAK (PASS)' if pass_vram else 'GAGAL'}** ({res2['vram_gb']:.3f} GB)

---

## 3. Kesimpulan & Keputusan Gerbang Latensi

{'**GERBANG TERBUKA!** Arsitektur Slim-Neck pada engine TensorRT FP16 terbukti memberikan efisiensi nyata pada hardware dan memenuhi kriteria real-time JRTIP. Anda sah secara akademis untuk melanjutkan ke fase training 100 epoch.' if pass_lat and pass_vram else '**REKOMENDASI OPREK ARSITEKTUR:** Model Slim-Neck membutuhkan penyesuaian lanjutan (misal eliminasi channel-shuffle atau lite-head) sebelum melangkah ke training 100 epoch.'}
"""
    with open('report_fase0_tensorrt.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("\n=================================================================")
    print("HASIL LENGKAP TERCATAT:")
    print(f"- Baseline TensorRT FP16 : {res1['mean_lat']:.2f} ms ({res1['fps']:.1f} FPS)")
    print(f"- Slim-Neck TensorRT FP16: {res2['mean_lat']:.2f} ms ({res2['fps']:.1f} FPS)")
    print(f"- VRAM Peak : {res2['vram_gb']:.3f} GB")
    print(f"- Report tersimpan di: report_fase0_tensorrt.md")
    print("=================================================================")

if __name__ == '__main__':
    main()

