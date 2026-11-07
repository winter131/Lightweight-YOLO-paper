# Light-YOLO: Real-Time & Efficient Underwater Object Detection
## Dokumen Investigasi Arsitektur, Fase 0 (Gerbang Latensi), dan Roadmap JRTIP Q1

Dokumen ini merangkum **status investigasi arsitektur terbaru**, **hasil pengujian empiris Fase 0 (Gerbang Latensi)**, dan **roadmap persiapan manuskrip** untuk jurnal internasional bereputasi: **Journal of Real-Time Image Processing (JRTIP) - Springer (Q1)**.

---

## 1. Fokus & Standar Kritis JRTIP (Springer Q1)

Sesuai ketentuan resmi editorial *Journal of Real-Time Image Processing*:
> *"To fall within the real-time scope of the journal, submitted manuscripts must address and discuss real-time issues for review consideration: computational complexity, frame processing rate of operation, real-time hardware implementation, and real-time software optimization."*

Target utama penelitian ini dialihkan ke skala **YOLOv11s** dengan target empiris:
1. **Target Latensi Inferensi:** $\le 4.5\text{ ms}$ ($\ge 220\text{ FPS}$) pada resolusi $640 \times 640$.
2. **Target Memory Footprint:** VRAM $\le 0.8\text{ GB}$ (ramah perangkat *embedded* AUV/ROV).
3. **Dataset Standar Internasional:** **URPC 2020** (16.991 citra) + **RUOD / UTDAC** *(Dataset Brackish resmi dihapus karena dinilai jenuh $\sim 98.9\%$)*.

---

## 2. Fase 0: "Gerbang Latensi" (*Pre-Flight Empirical Check*)

Sebelum menjalankan pelatihan 100 epoch yang memakan waktu dan komputasi GPU berhari-hari, dilakukan **Pre-Flight Check** pada model mentah (*untrained / random weights*) untuk menguji apakah modifikasi *Slim-Neck* secara fisik memberikan akselerasi nyata pada perangkat keras.

### Setup Pengujian Fase 0:
* **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (TGP 75W, CUDA 11.8)
* **Kondisi:** Batch size = 1, Interleaved A/B Benchmark (100 run bergantian), Fused Layers, FP16 Half Precision.
* **Model yang Diuji:**
  * **State 1:** Baseline YOLOv11s (Standar C3k2 + Conv Neck)
  * **State 2:** YOLOv11s + Slim-Neck (`GSConv` + `VoVGSCSP` Neck)

---

### Hasil Empiris Pengujian Fase 0 (YOLOv11s):

| Metrik Evaluasi | Baseline YOLOv11s (State 1) | YOLOv11s + Slim-Neck (State 2) | Selisih / Efek Modifikasi | Status Gerbang JRTIP |
| :--- | :---: | :---: | :---: | :---: |
| **Jumlah Lapisan (*Layers*)** | **99 layers** | 104 layers | +5 layers (Fragmentasi meningkat) | - |
| **Parameter Jaringan (M)** | 9.33 M | **8.28 M** | **-11.3% Parameter (-1.05M)** | ✅ **LOLOS** |
| **Beban Komputasi (FLOPs)** | 17.7 GFLOPs | **16.0 GFLOPs** | **-9.6% FLOPs (-1.70G)** | ✅ **LOLOS** |
| **Peak VRAM Footprint** | 0.039 GB (39.7 MB) | **0.054 GB (55.5 MB)** | Sangat hemat memori | 🏆 **LOLOS TELAK ($\le 0.8\text{ GB}$)** |
| **Latensi $640 \times 640$ (Interleaved)** | **16.05 ms** (62.3 FPS) | **17.22 ms** (58.1 FPS) | **+1.17 ms (-7.3% Lebih Lambat)** | ❌ **BELUM TEMBUS ($\le 4.5\text{ ms}$)** |
| **Latensi $512 \times 512$** | **16.87 ms** (59.3 FPS) | **17.46 ms** (57.3 FPS) | **+0.59 ms (-3.5% Lebih Lambat)** | ❌ **BELUM TEMBUS ($\le 4.5\text{ ms}$)** |
| **Latensi $480 \times 480$** | **17.00 ms** (58.8 FPS) | **17.53 ms** (57.0 FPS) | **+0.53 ms (-3.1% Lebih Lambat)** | ❌ **BELUM TEMBUS ($\le 4.5\text{ ms}$)** |
| **Latensi $416 \times 416$** | **15.78 ms** (63.4 FPS) | **16.77 ms** (59.6 FPS) | **+0.99 ms (-6.3% Lebih Lambat)** | ❌ **BELUM TEMBUS ($\le 4.5\text{ ms}$)** |

---

## 3. Bedah Mekanistik: Kenapa Slim-Neck Malah 7% Lebih Lambat di PyTorch?

Hasil empiris Fase 0 membuktikan teori arsitektur komputer (*Roofline Model*):

```
                   Analisis Overhead GSConv pada Batch 1
1x Standard Conv ($3 \times 3$)  ──► 1 CUDA Kernel Launch ──► Eksekusi GPU Padat (Compute-Bound)
GSConv Block              ──► 5 Operasi Terpisah:
                              ├── 1x Conv $1 \times 1$
                              ├── 1x Depthwise Conv $5 \times 5$
                              ├── 1x Concat (`torch.cat`)
                              └── 1x Channel Shuffle (`permute + contiguous`)
                              (Memory-Bound & Kernel Launch Overhead mendominasi!)
```

1. **Penalti Fragmentasi Kernel:**
   Meskipun perkalian matematika (FLOPs) berkurang 9.6%, penambahan 5 layer pecahan menyebabkan GPU menghabiskan waktu lebih banyak untuk meluncurkan kernel dan mengacak alamat memori DRAM (*memory shuffle*).
2. **Kesimpulan Gerbang:**
   **GERBANG LATENSI DITAHAN.** Pelatihan 100 epoch untuk State 3 dst. **TIDAK DILANJUTKAN** sebelum arsitektur Slim-Neck dimodifikasi (*di-oprek*) hingga terbukti lebih cepat secara deterministik.

---

## 4. Rencana Oprek Arsitektur Menembus $\le 4.5\text{ ms}$

Untuk menembus batas $\le 4.5\text{ ms}$, disiapkan 3 alternatif modifikasi struktur:

1. **Eliminasi `Channel Shuffle`:**
   Mengganti GSConv dengan **Depthwise Separable Conv (DSC) murni** atau **GhostConv tanpa shuffle** untuk melenyapkan *memory traffic bottleneck*.
2. **Pemangkasan Kedalaman Neck (*Neck Pruning*):**
   Mengurangi repetisi blok integrasi fitur di Neck dari $n=2$ menjadi $n=1$.
3. **Penyederhanaan Detection Head (*Shared / Lite Head*):**
   Menyederhanakan lapisan konvolusi pada *Decoupled Head* yang merupakan penyumbang latensi terbesar pada keluarga YOLOv11.

---

## 5. Status Modul Lanjutan (Dalam Penelusuran Ulang)

* **State 3 (Scheduled IoG-Gated Repulsion Loss):**
  * Terbukti secara matematis menyembuhkan *gradient explosion* (tanpa *crash NaN*) dan mendongkrak oklusi kelas *Starfish* (+0.82% pada pengujian awal).
  * Akan diintegrasikan kembali setelah arsitektur Neck yang lolos Gerbang Latensi final ditetapkan.
* **State 4 (Class-Balanced Loss):**
  * Menggunakan inverse effective number of samples untuk penyeimbangan kelas minoritas (*Holothurian*). Ditahan untuk pengujian bertahap pasca-Gerbang Latensi.
* **Benchmark Target:**
  * Dataset: **URPC 2020** + **RUOD / UTDAC**.
  * Pipeline NMS: Diuji dengan NMS seragam untuk menjamin perbandingan arsitektur yang adil (*apple-to-apple*).
