import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from scipy.optimize import linear_sum_assignment

# ===========================
# 1. Anchor Generator (教程版)
# ===========================
def make_anchors(height, width, stride=8, device="cpu"):
    Hf = height // stride
    Wf = width  // stride

    # Centers of each 8x8 patch
    ys = (torch.arange(Hf, device=device) + 0.5) * stride
    xs = (torch.arange(Wf, device=device) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij") 

    # N=4 anchors per grid cell
    grid_x = grid_x.unsqueeze(0).repeat(4, 1, 1)
    grid_y = grid_y.unsqueeze(0).repeat(4, 1, 1)
    
    # 【关键】Delta 偏移 (教程必须项)
    delta = stride // 4
    grid_x[[0, 2]] -= delta
    grid_x[[1, 3]] += delta
    grid_y[[0, 1]] -= delta
    grid_y[[2, 3]] += delta
    
    anchors = torch.stack([grid_x, grid_y], dim=0)  # (2, N, Hf, Wf)
    return anchors

# ===========================
# 2. Backbone (Truncated VGG16)
# ===========================
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练权重 (兼容旧版 torchvision)
        vgg = vgg16(pretrained=True)
        features = list(vgg.features.children())
        
        # Block 4 Output: index 0 to 22 (ReLU after Conv4_3)
        self.block4 = nn.Sequential(*features[:23])
        # Block 5 Output: index 23 to 29 (ReLU after Conv5_3)
        self.block5 = nn.Sequential(*features[23:30])
        
        # 冻结参数逻辑会在 train.py 中控制，这里默认 requires_grad=True

    def forward(self, x):
        c4 = self.block4(x) # H/8
        c5 = self.block5(c4) # H/16
        return c4, c5

# ===========================
# 3. P2PNet Complete Model
# ===========================
class P2PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        
        # Neck
        self.neck_c4 = nn.Conv2d(512, 256, 1)
        self.neck_c5 = nn.Conv2d(512, 256, 1)
        self.neck_refine = nn.Conv2d(256, 256, 3, padding=1)

        # Heads (N=4)
        self.head_reg = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 4 * 2, 3, padding=1) # 2N coordinates
        )
        self.head_cls = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 4, 3, padding=1)     # N scores
        )

    def forward(self, x):
        c4, c5 = self.backbone(x)
        
        # Neck FPN
        c5_reduced = self.neck_c5(c5)
        c4_reduced = self.neck_c4(c4)
        
        # Upsample C5 to match C4 size
        c5_up = F.interpolate(c5_reduced, size=c4_reduced.shape[2:], mode='nearest')
        feature = self.neck_refine(c4_reduced + c5_up)
        
        # Heads
        # Regression: [B, 8, H/8, W/8]
        out_reg = self.head_reg(feature)
        # Classification: [B, 4, H/8, W/8]
        out_cls = self.head_cls(feature)
        
        # Decode points
        B, _, Hf, Wf = out_cls.shape
        anchors = make_anchors(x.shape[2], x.shape[3], stride=8, device=x.device)
        
        # Reshape to [B, N*Hf*Wf, 2] and [B, N*Hf*Wf, 1]
        anchors = anchors.view(2, -1).permute(1, 0).unsqueeze(0).repeat(B, 1, 1) #(B, NumPoints, 2)
        
        pred_offsets = out_reg.view(B, 2, -1).permute(0, 2, 1) # (B, NumPoints, 2)
        pred_points = anchors + pred_offsets # Apply offsets
        
        pred_logits = out_cls.view(B, -1, 1) # (B, NumPoints, 1)
        
        return {'pred_logits': pred_logits, 'pred_points': pred_points}

# ===========================
# 4. Matcher & Loss (严格按照教程)
# ===========================
class HungarianMatcher(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions
        out_prob = outputs['pred_logits'].sigmoid().flatten(0, 1).squeeze(-1) # [batch*queries]
        out_bbox = outputs['pred_points'].flatten(0, 1) # [batch*queries, 2]
        
        indices = []
        for i in range(bs):
            # 拿到第 i 张图的预测和 GT
            p_prob = out_prob[i*num_queries : (i+1)*num_queries]
            p_bbox = out_bbox[i*num_queries : (i+1)*num_queries]
            tgt_bbox = targets[i]['point'].to(p_bbox.device)
            
            if len(tgt_bbox) == 0:
                empty = torch.as_tensor([], dtype=torch.int64)
                indices.append((empty, empty))
                continue

            # Compute Cost Matrix
            # Cost = tau * Dist + (1 - Score)
            # Distance (L2)
            dists = torch.cdist(p_bbox, tgt_bbox, p=2) # [num_queries, num_gt]
            
            # Score Cost
            # shape matching: p_prob [Q], tgt [G] -> need [Q, G]
            # p_prob.unsqueeze(-1) gives [Q, 1]
            cost_class = 1 - p_prob.unsqueeze(-1)
            
            C = self.tau * dists + cost_class
            
            # Hungarian Algorithm
            C_cpu = C.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(C_cpu)
            
            indices.append((torch.as_tensor(row_ind, dtype=torch.int64), 
                            torch.as_tensor(col_ind, dtype=torch.int64)))
        
        return indices

class P2PNetLoss(nn.Module):
    def __init__(self, matcher, lambda_neg=0.5, use_focal=False, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.matcher = matcher
        self.lambda_neg = lambda_neg # 教程中的 lambda_neg
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        # These weights mirror DETR-style balancing between classification and point losses
        self.weight_dict = {
            'loss_ce': 1.0,
            'loss_point': 1.0,
        }

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        
        # Helper to index tensors
        idx = self._get_src_permutation_idx(indices)
        src_logits = outputs['pred_logits']
        src_points = outputs['pred_points']

        matched_points = [t['point'][i] for t, (_, i) in zip(targets, indices) if i.numel() > 0]
        if matched_points:
            target_points = torch.cat(matched_points, dim=0)
        else:
            target_points = torch.zeros((0, 2), device=src_points.device, dtype=src_points.dtype)

        # --- 1. Classification Loss (BCE or Focal) ---
        target_classes = torch.zeros_like(src_logits.squeeze(-1))
        target_classes[idx] = 1.0 # Matched = 1

        logits = src_logits.squeeze(-1)
        if self.use_focal:
            probs = torch.sigmoid(logits)
            pt = probs * target_classes + (1 - probs) * (1 - target_classes)
            alpha_t = self.focal_alpha * target_classes + (1 - self.focal_alpha) * (1 - target_classes)
            focal_weight = alpha_t * (1 - pt).pow(self.focal_gamma)
            bce = F.binary_cross_entropy_with_logits(logits, target_classes, reduction='none')
            neg_weight = torch.ones_like(target_classes)
            neg_weight[target_classes < 0.5] = self.lambda_neg
            loss_ce = (focal_weight * bce * neg_weight).mean()
        else:
            # Weights: matched=1.0, background=lambda_neg
            weights = torch.full_like(logits, self.lambda_neg)
            weights[idx] = 1.0
            loss_ce = F.binary_cross_entropy_with_logits(
                logits,
                target_classes,
                weight=weights
            )

        # --- 2. Localization Loss (MSE on matched only) ---
        if len(target_points) == 0:
            loss_point = src_points.sum() * 0
        else:
            loss_point = F.mse_loss(src_points[idx], target_points)

        return {'loss_ce': loss_ce, 'loss_point': loss_point}

    def _get_src_permutation_idx(self, indices):
        batch_entries = []
        src_entries = []
        for i, (src, _) in enumerate(indices):
            if src.numel() == 0:
                continue
            batch_entries.append(torch.full_like(src, i, dtype=torch.int64))
            src_entries.append(src.to(torch.int64))

        if not batch_entries:
            empty = torch.as_tensor([], dtype=torch.int64)
            return empty, empty

        batch_idx = torch.cat(batch_entries)
        src_idx = torch.cat(src_entries)
        return batch_idx, src_idx