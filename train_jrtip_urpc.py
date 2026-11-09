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
    parser = argparse.ArgumentParser(description="JRTIP URPC 2020 Ablation Training")
    parser.add_argument("--state", type=int, default=1, help="State number (1: Baseline, 2: Slim-Neck, etc.)")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs to train (default: 2 for pre-check)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (default: 0)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # --- PENGATURAN DISIPLIN JURNAL Q1 JRTIP ---
    # 1. Matikan fitur Auto-Magic & Kunci Hyperparameter
    lr0 = 0.01
    cos_lr = False
    patience = 50
    seed = 42
    deterministic = True
    
    # 2. Pemilihan Model Config sesuai State
    if args.state == 1:
        cfg_file = "configs/yolo11s_state1.yaml"
        run_name = "State1_Baseline_v11s"
        custom_modules.USE_REPULSION_LOSS = False
        custom_modules.USE_CLASS_BALANCED = False
    elif args.state == 2:
        cfg_file = "configs/yolo11s_state2.yaml"
        run_name = "State2_SlimNeck_v11s"
        custom_modules.USE_REPULSION_LOSS = False
        custom_modules.USE_CLASS_BALANCED = False
    elif args.state == 3:
        cfg_file = "configs/yolo11s_state1.yaml"
        run_name = "State3_RepulsionOnly"
        custom_modules.USE_REPULSION_LOSS = True
        custom_modules.USE_CLASS_BALANCED = False
    elif args.state == 4:
        cfg_file = "configs/yolo11s_state1.yaml"
        run_name = "State4_ClassBalancedOnly"
        custom_modules.USE_REPULSION_LOSS = False
        custom_modules.USE_CLASS_BALANCED = True
    elif args.state == 8:
        cfg_file = "configs/yolo11s_state2.yaml"
        run_name = "State8_Full_v11s"
        custom_modules.USE_REPULSION_LOSS = True
        custom_modules.USE_CLASS_BALANCED = True
    else:
        raise ValueError(f"State {args.state} belum didefinisikan.")

    print("\n" + "="*65)
    print(f"JRTIP SPRINGER (Q1) - URPC 2020 REPRODUCIBLE ABLATION")
    print(f"-> State        : {args.state} ({run_name})")
    print(f"-> Arsitektur   : {cfg_file}")
    print(f"-> Target Epoch : {args.epochs}")
    print(f"-> Batch Size   : {args.batch} (Manual Lock - No Auto-Batch)")
    print(f"-> Seed         : {seed} (Deterministic: {deterministic})")
    print(f"-> Learning Rate: lr0={lr0} (cos_lr={cos_lr})")
    print(f"-> Project Dir  : JRTIP_URPC_Ablation/{run_name}")
    print("="*65 + "\n")

    # Inisialisasi Model
    project_dir = "JRTIP_URPC_Ablation"
    ckpt_path = os.path.join("runs", "detect", project_dir, run_name, "weights", "last.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(project_dir, run_name, "weights", "last.pt")
    
    custom_modules.CURRENT_EPOCH = 1
    custom_modules.MAX_EPOCHS = args.epochs

    if args.resume and os.path.exists(ckpt_path):
        print(f"[*] Resuming from checkpoint: {ckpt_path}")
        model = YOLO(ckpt_path)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        train_results = model.train(
            data="dataset/urpc_raw/data.yaml",
            epochs=args.epochs,
            resume=True
        )
    else:
        print(f"[*] Starting fresh training from: {cfg_file}")
        model = YOLO(cfg_file)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        train_results = model.train(
            data="dataset/urpc_raw/data.yaml",
            epochs=args.epochs,
            patience=patience,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=lr0,
            cos_lr=cos_lr,
            seed=seed,
            deterministic=deterministic,
            project=project_dir,
            name=run_name,
            exist_ok=True,
            save=True,
            save_period=1,
            device=args.device,
            workers=4,
            plots=True,
            verbose=True
        )

    print("\n" + "="*65)
    print(f"TRAINING {args.epochs} EPOCH SELESAI UNTUK {run_name}!")
    print(f"Checkpoint tersimpan di: {os.path.join('runs', 'detect', project_dir, run_name, 'weights')}")
    print("="*65 + "\n")

    best_pt = os.path.join("runs", "detect", project_dir, run_name, "weights", "best.pt")
    if os.path.exists(best_pt):
        print(f"\n[*] Evaluating best model: {best_pt}")
        eval_model = YOLO(best_pt)
        metrics = eval_model.val(
            data="dataset/urpc_raw/data.yaml",
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device
        )
        print("\n" + "="*65)
        print(f"HASIL FINAL EVALUASI ({run_name}) - best.pt:")
        print(f"Overall mAP50    : {metrics.box.map50:.4f}")
        print(f"Overall mAP50-95 : {metrics.box.map:.4f}")
        print(f"Precision        : {metrics.box.mp:.4f}")
        print(f"Recall           : {metrics.box.mr:.4f}")
        names = ['echinus', 'holothurian', 'scallop', 'starfish']
        for i, name in enumerate(names):
            if i < len(metrics.box.maps):
                print(f"Class {name:12}: mAP50 = {metrics.box.maps[i]:.4f}")
        print("="*65 + "\n")

if __name__ == "__main__":
    main()

