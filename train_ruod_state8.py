import os
import argparse
import torch
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Register custom modules to allow parse_model to recognize custom classes
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

def on_train_epoch_start(trainer):
    custom_modules.CURRENT_EPOCH = trainer.epoch + 1

def main():
    print("=" * 65)
    print("🚀 MEMULAI TRAINING DATASET RUOD (10 KELAS) DENGAN STATE 8...")
    print("=" * 65)

    # 1. Arsitektur Fisik Slim-Neck (Reset bobot dari nol untuk perbandingan adil)
    cfg_file = "yolo11s-slimneck.yaml"
    project_dir = "JRTIP_Generalization"
    run_name = "RUOD_State8_SlimNeck_Rep_CB"

    # 2. Senjata Utama Zero-Inference-Cost (State 8)
    custom_modules.USE_REPULSION_LOSS = True   # Sumbu B: Optimasi Oklusi Bawah Air
    custom_modules.USE_CLASS_BALANCED = True    # Sumbu C: Kompensasi Ketimpangan Kelas
    custom_modules.CURRENT_EPOCH = 1
    custom_modules.MAX_EPOCHS = 100

    # 3. Hiperparameter Identik & Disiplin JRTIP Springer Q1
    epochs = 100
    batch = 16
    imgsz = 640
    patience = 50
    seed = 42
    deterministic = True
    lr0 = 0.01
    cos_lr = False

    # Check for resume
    ckpt_path = os.path.join("runs", "detect", project_dir, run_name, "weights", "last.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(project_dir, run_name, "weights", "last.pt")

    if os.path.exists(ckpt_path):
        print(f"[*] Resuming from checkpoint: {ckpt_path}")
        model = YOLO(ckpt_path)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        results = model.train(
            data="dataset/ruod/data.yaml",
            epochs=epochs,
            resume=True
        )
    else:
        print(f"[*] Starting fresh training from scratch: {cfg_file}")
        model = YOLO(cfg_file)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        results = model.train(
            data="dataset/ruod/data.yaml",
            epochs=epochs,
            patience=patience,
            batch=batch,
            imgsz=imgsz,
            lr0=lr0,
            cos_lr=cos_lr,
            seed=seed,
            deterministic=deterministic,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=1,
            device=0
        )

    print("\n" + "=" * 65)
    print("✅ TRAINING RUOD SELESAI!")
    print("=" * 65)

if __name__ == "__main__":
    main()
