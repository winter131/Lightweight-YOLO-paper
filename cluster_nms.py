import torch
import torchvision
import time
import math
from ultralytics.utils import ops

def cluster_nms_with_spm(boxes, scores, iou_threshold=0.5, penalty_factor=0.2):
    """
    Cluster-NMS with Score Penalty Mechanism (SPM).
    Ref: CGC-YOLO (Z. Zhang et al., 2024)
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
    # Sort boxes by scores in descending order
    sorted_scores, order = scores.sort(0, descending=True)
    sorted_boxes = boxes[order]
    
    N = sorted_boxes.shape[0]
    # Calculate IoU matrix
    ious = torchvision.ops.box_iou(sorted_boxes, sorted_boxes)
    # Upper triangular matrix with zeros on diagonal
    ious = torch.triu(ious, diagonal=1)
    
    # Cluster-NMS iteration
    b = torch.ones(N, dtype=torch.float, device=boxes.device)
    T = N
    for i in range(T):
        A = torch.diag(b)
        C = torch.mm(A, ious)
        
        # Find maximum for each column
        g = C.max(dim=0)[0]
        
        # Binarize
        b_new = (g < iou_threshold).float()
        
        if torch.equal(b, b_new):
            C_final = C
            break
        b = b_new
    else:
        C_final = torch.mm(torch.diag(b), ious)
        
    # Score Penalty Mechanism (SPM)
    # Penalize scores of boxes that heavily overlap with the preserved ones
    # Only impose penalty if IoU > threshold
    # Tuned SPM parameters to prevent duplicate box flooding under max_det=300
    # penalty_factor (sigma) is set smaller (0.05) to steepen the penalty curve
    C_mask = (C_final > iou_threshold).float()
    penalty_terms = torch.exp(-(C_final * C_mask)**2 / 0.05)
    
    # Product of penalty terms for each box
    penalties = penalty_terms.prod(dim=0)
    
    # Apply penalty to scores
    penalized_scores = sorted_scores * penalties
    
    # Keep boxes with scores above a higher tuned threshold after penalty
    keep_mask = penalized_scores > 0.005
    
    valid_scores = penalized_scores[keep_mask]
    valid_order = order[keep_mask]
    
    # Re-sort remaining valid boxes by their new penalized scores
    _, new_sort_idx = valid_scores.sort(0, descending=True)
    return valid_order[new_sort_idx]


def custom_non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
        **kwargs
):
    """
    Overridden non_max_suppression from Ultralytics using Cluster-NMS + SPM.
    """
    # Standard YOLO output processing before NMS
    # Shape: (batch, num_classes + 4, num_anchors)
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    t = time.time()
    
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        box = ops.xywh2xyxy(box)  # center_x, center_y, width, height) to (x1, y1, x2, y2)

        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # APLLY CLUSTER NMS HERE
        i = cluster_nms_with_spm(boxes, scores, iou_threshold=iou_thres, penalty_factor=0.2)
        
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        
        if (time.time() - t) > time_limit:
            break

    return output

# Monkey patching function
import ultralytics.utils.nms as yolo_nms
_original_nms = yolo_nms.non_max_suppression

def patch_nms():
    yolo_nms.non_max_suppression = custom_non_max_suppression

def unpatch_nms():
    yolo_nms.non_max_suppression = _original_nms
