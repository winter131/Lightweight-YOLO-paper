import sys
sys.path.insert(0, '.')
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_modules.BaselineBlock = custom_modules.BaselineBlock

yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.BaselineBlock = custom_modules.BaselineBlock

custom_modules.USE_CLASS_BALANCED = True
custom_modules.USE_REPULSION_LOSS = True

def on_train_epoch_start(trainer):
    custom_modules.CURRENT_EPOCH = trainer.epoch + 1

def main():
    print("==================================================")
    print("RESUMING BRACKISH STATE 8 TRAINING FROM LAST CHECKPOINT")
    print("==================================================")
    model = YOLO(r"runs\detect\Brackish_state8\weights\last.pt")
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.train(resume=True)

if __name__ == '__main__':
    main()
