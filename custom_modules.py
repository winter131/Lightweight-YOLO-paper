import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, C3k2, Bottleneck

class GSConv(nn.Module):
    """
    GSConv: Ghost-Shuffle Convolution for Slim-Neck design.
    Combines standard convolution, depth-wise convolution, and channel shuffle.
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        c_ = max(1, int(c2 // 2))
        self.cv1 = Conv(c1, c_, k, s, p, g=g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, 2, g=c_, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        
        # Channel shuffle
        b, c, h, w = x2.size()
        b_n, c_n = b, c // 2
        x2 = x2.view(b_n, 2, c_n, h, w)
        x2 = torch.transpose(x2, 1, 2).contiguous()
        x2 = x2.view(b_n, c, h, w)
        return x2

class GSBottleneck(nn.Module):
    """
    GSBottleneck: Ghost-Shuffle Bottleneck.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = max(1, int(c2 * e))
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = GSConv(c_, c2, 3, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class VoVGSCSP(nn.Module):
    """
    VoVGSCSP module for Slim-Neck.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        x2 = self.cv2(x)
        return self.cv3(torch.cat((x1, x2), 1))

class BaselineBlock(nn.Module):
    """
    A pure neutral non-attention block to replace C2PSA.
    We use standard C3k2 without any attention mechanism for an unbiased baseline.
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.block = C3k2(c1, c2, n, shortcut, g, e)

    def forward(self, x):
        return self.block(x)

# --- ABLATION STATE FLAGS & DECAY TRACKING ---
USE_REPULSION_LOSS = False
USE_CLASS_BALANCED = False
CURRENT_EPOCH = 1
MAX_EPOCHS = 100

# --- CUSTOM LOSS FUNCTION ---
from ultralytics.utils.loss import v8DetectionLoss, FocalLoss
from ultralytics.utils.tal import make_anchors

def custom_get_assigned_targets_and_loss(self, preds, batch):
    """Calculate the sum of the loss for box, cls and dfl multiplied by batch size and return foreground mask and target indices."""
    loss = torch.zeros(3, device=self.device)  # box, cls, dfl
    pred_distri, pred_scores = (
        preds["boxes"].permute(0, 2, 1).contiguous(),
        preds["scores"].permute(0, 2, 1).contiguous(),
    )
    anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)

    dtype = pred_scores.dtype
    batch_size = pred_scores.shape[0]
    imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

    # Targets
    targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
    targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
    gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
    mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

    # Pboxes
    pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

    _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
        pred_scores.detach().sigmoid(),
        (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
        anchor_points * stride_tensor,
        gt_labels,
        gt_bboxes,
        mask_gt,
    )

    target_scores_sum = max(target_scores.sum(), 1)

    # ---------------------------------------------------------
    # AXIS C: CLASS-BALANCED LOSS
    # ---------------------------------------------------------
    global USE_CLASS_BALANCED
    if USE_CLASS_BALANCED:
        # Hitung jumlah kemunculan kelas dalam batch ini
        counts = torch.bincount(batch["cls"].view(-1).long(), minlength=self.nc)
        counts = torch.clamp(counts, min=1.0)
        
        # Inverse frequency: kelas yg sering muncul dpt bobot kecil
        alpha = 1.0 / counts
        alpha = (alpha / alpha.mean()).to(device=self.device, dtype=dtype) # Normalisasi agar rata-ratanya 1
        
        # Hitung BCE Loss (shape: batch, anchors, nc)
        bce_out = self.bce(pred_scores, target_scores.to(dtype))
        # Kalikan dengan bobot kelas (broadcast)
        bce_out = bce_out * alpha.view(1, 1, self.nc)
        
        loss[1] = bce_out.sum() / target_scores_sum
    else:
        if getattr(self.hyp, "fl_gamma", 0.0) > 0.0:
            if not hasattr(self, "focal_loss"):
                self.focal_loss = FocalLoss(gamma=self.hyp.fl_gamma)
            loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)) / target_scores_sum
        else:
            loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

    # Bbox loss
    if fg_mask.sum():
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        
        # ---------------------------------------------------------
        # AXIS B: DYNAMIC SCHEDULED IOG-GATED REPULSION LOSS
        # ---------------------------------------------------------
        global USE_REPULSION_LOSS, CURRENT_EPOCH, MAX_EPOCHS
        if USE_REPULSION_LOSS:
            import math
            from torchvision.ops import box_iou
            b_sz = pred_bboxes.shape[0]
            loss_rep_gt = torch.tensor(0.0, device=self.device)
            loss_rep_box = torch.tensor(0.0, device=self.device)
            num_pos = 0.0
            sigma = 0.5
            
            # Dynamic Cosine Decay: Active (1.0) for Epochs 1-50, Cosine Decay (1.0 -> 0.0) for 51-75, OFF (0.0) for 76-100
            if CURRENT_EPOCH <= 50:
                decay_factor = 1.0
            elif CURRENT_EPOCH <= 75:
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * ((CURRENT_EPOCH - 50) / 25.0)))
            else:
                decay_factor = 0.0
                
            if decay_factor > 0.0:
                for i in range(b_sz):
                    fg = fg_mask[i]
                    if not fg.any(): continue
                    
                    # Prediksi positif dan GT asli mereka
                    preds = pred_bboxes[i][fg] # (N, 4)
                    targets_idx = target_gt_idx[i][fg].long() # (N,)
                    
                    valid_gts = mask_gt[i]
                    N = preds.shape[0]
                    num_pos += N
                    
                    # --- RepGT: IoG-Gated (IoG > 0.15) ---
                    if valid_gts.any():
                        gts_full = gt_bboxes[i] # (max_gt, 4)
                        
                        # Manual intersection untuk IoG
                        lt_f = torch.max(preds[:, None, :2], gts_full[None, :, :2])
                        rb_f = torch.min(preds[:, None, 2:], gts_full[None, :, 2:])
                        wh_f = (rb_f - lt_f).clamp(min=0)
                        inter_f = wh_f[:, :, 0] * wh_f[:, :, 1]
                        area_gts_f = (gts_full[:, 2] - gts_full[:, 0]) * (gts_full[:, 3] - gts_full[:, 1])
                        iog_f = inter_f / area_gts_f[None, :].clamp(min=1e-6) # (N, max_gt)
                        
                        # Masking GT invalid (padding)
                        valid_mask = mask_gt[i].squeeze(-1).unsqueeze(0).expand(N, -1).bool()
                        iog_f[~valid_mask] = -1.0
                        
                        # Masking GT asli (jangan di-repel dari GT-nya sendiri)
                        batch_idx = torch.arange(N, device=self.device)
                        iog_f[batch_idx, targets_idx] = -1.0
                        
                        # Cari GT lain yang paling nempel (max IoG)
                        max_iog, _ = iog_f.max(dim=1)
                        
                        # IoG Gate: Penalize only overlapping instances IoG > 0.15
                        gated_gt_mask = max_iog > 0.15
                        if gated_gt_mask.any():
                            max_iog_gated = torch.clamp(max_iog[gated_gt_mask], 0.15, 0.999)
                            # Smooth-Ln L_RepGT (Wang et al. 2018, CVPR)
                            mask_gt_sigma = max_iog_gated <= sigma
                            l_rep_gt = torch.empty_like(max_iog_gated)
                            l_rep_gt[mask_gt_sigma] = -torch.log(1.0 - max_iog_gated[mask_gt_sigma] + 1e-6)
                            l_rep_gt[~mask_gt_sigma] = (max_iog_gated[~mask_gt_sigma] - sigma) / (1.0 - sigma) - torch.log(torch.tensor(1.0 - sigma + 1e-6, device=self.device))
                            loss_rep_gt += l_rep_gt.sum()
                        
                    # --- RepBox: IoU-Gated (IoU > 0.15) ---
                    if N > 1:
                        iou_preds = box_iou(preds, preds) # (N, N)
                        # Masking pasangan milik GT yang sama (hanya repel dari objek beda)
                        same_gt_mask = targets_idx.unsqueeze(1) == targets_idx.unsqueeze(0)
                        iou_preds[same_gt_mask] = -1.0
                        valid_iou = iou_preds[~same_gt_mask]
                        
                        if valid_iou.numel() > 0:
                            gated_box_mask = valid_iou > 0.15
                            if gated_box_mask.any():
                                valid_iou_gated = torch.clamp(valid_iou[gated_box_mask], 0.15, 0.999)
                                # Smooth-Ln L_RepBox (Wang et al. 2018, CVPR)
                                mask_box_sigma = valid_iou_gated <= sigma
                                l_rep_box = torch.empty_like(valid_iou_gated)
                                l_rep_box[mask_box_sigma] = -torch.log(1.0 - valid_iou_gated[mask_box_sigma] + 1e-6)
                                l_rep_box[~mask_box_sigma] = (valid_iou_gated[~mask_box_sigma] - sigma) / (1.0 - sigma) - torch.log(torch.tensor(1.0 - sigma + 1e-6, device=self.device))
                                loss_rep_box += l_rep_box.sum()
                
                # Base weights 0.35 multiplied by dynamic decay factor
                alpha_rep = 0.35 * decay_factor
                beta_rep = 0.35 * decay_factor
                num_pos_clamp = max(1.0, float(num_pos))
                total_rep_loss = alpha_rep * (loss_rep_gt / num_pos_clamp) + beta_rep * (loss_rep_box / num_pos_clamp)

    loss[0] *= self.hyp.box  # box gain
    if USE_REPULSION_LOSS and fg_mask.sum() and 'total_rep_loss' in locals():
        # Inject Repulsion Loss as auxiliary term (not pre-multiplied by 7.5)
        loss[0] += total_rep_loss

    loss[1] *= self.hyp.cls  # cls gain
    loss[2] *= self.hyp.dfl  # dfl gain
    return (
        (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
        loss,
        loss.detach(),
    )

# Inject monkey patch method secara langsung ke class v8DetectionLoss
v8DetectionLoss.get_assigned_targets_and_loss = custom_get_assigned_targets_and_loss
