import sys
sys.path.insert(0, '.')
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP

# State 3 = Baseline + Repulsion Loss (no Slim-Neck, no Class-Balanced)
custom_modules.USE_CLASS_BALANCED = False
custom_modules.USE_REPULSION_LOSS = True

def main():
    print("==================================================")
    print("RESUMING STATE 3 TRAINING FROM LAST CHECKPOINT")
    print("==================================================")
    model = YOLO(r"runs\detect\URPC_state3\weights\last.pt")
    model.train(resume=True)

if __name__ == '__main__':
    main()
