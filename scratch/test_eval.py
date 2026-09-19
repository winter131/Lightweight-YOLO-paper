import os
import sys
sys.path.insert(0, '.')
from ultralytics import YOLO
import custom_modules
import ultralytics.nn.modules as yolo_modules
import ultralytics.nn.tasks as yolo_tasks
from cluster_nms import patch_nms

yolo_modules.GhostConv = custom_modules.GSConv
yolo_modules.C3Ghost = custom_modules.VoVGSCSP
yolo_tasks.GhostConv = custom_modules.GSConv
yolo_tasks.C3Ghost = custom_modules.VoVGSCSP

def main():
    model = YOLO(r'runs/detect/URPC_state8/weights/best.pt')

    print('--- Evaluating best.pt with imgsz=640 ---')
    res640 = model.val(data='dataset/urpc_raw/data.yaml', imgsz=640, device='0', workers=0)
    print(f'imgsz=640 mAP50: {res640.results_dict["metrics/mAP50(B)"]*100:.2f}%')
    print(f'imgsz=640 mAP50-95: {res640.results_dict["metrics/mAP50-95(B)"]*100:.2f}%')

    print('\n--- Evaluating best.pt with imgsz=800 ---')
    res800 = model.val(data='dataset/urpc_raw/data.yaml', imgsz=800, device='0', workers=0)
    print(f'imgsz=800 mAP50: {res800.results_dict["metrics/mAP50(B)"]*100:.2f}%')
    print(f'imgsz=800 mAP50-95: {res800.results_dict["metrics/mAP50-95(B)"]*100:.2f}%')

    print('\n--- Evaluating best.pt with Cluster-NMS + imgsz=800 ---')
    patch_nms()
    res800_nms = model.val(data='dataset/urpc_raw/data.yaml', imgsz=800, device='0', workers=0)
    print(f'Cluster-NMS imgsz=800 mAP50: {res800_nms.results_dict["metrics/mAP50(B)"]*100:.2f}%')
    print(f'Cluster-NMS imgsz=800 mAP50-95: {res800_nms.results_dict["metrics/mAP50-95(B)"]*100:.2f}%')

if __name__ == '__main__':
    main()
