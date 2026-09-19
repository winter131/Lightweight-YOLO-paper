from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks

# Inject custom modules so YOLO parses them from the YAML / PT correctly
yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP

def resume_training():
    print("==================================================")
    print("RESUMING STATE 8 TRAINING FROM LAST CHECKPOINT")
    print("==================================================")
    
    # Load model dari checkpoint terakhir
    model = YOLO(r"runs\detect\URPC_state8\weights\last.pt")
    
    # Lanjutkan training dengan konfigurasi yang sama
    model.train(resume=True)

if __name__ == '__main__':
    resume_training()
