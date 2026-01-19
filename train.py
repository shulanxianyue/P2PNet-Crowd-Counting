import torch
import os
import time
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from dataset import CrowdDataset, collate_fn
from p2pnet import P2PNet, P2PNetLoss, HungarianMatcher

# --- Config ---
DATA_ROOT = "expd/ShanghaiTech"
EPOCHS = 500
BATCH_SIZE = 8  # 如果显存不够(报错OOM)，把这里改成 2 或 1
VAL_BATCH_SIZE = 1
LR = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DIR = "weights"
RESULTS_DIR = "results_train"
DEVICE = "cuda"  # <--- 关键补充

# Loss weight for negatives
LAMBDA_NEG = 2.0

# --- Focal Loss ---
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# --- Scheduler ---
SCHEDULER_STEP = 50
SCHEDULER_GAMMA = 0.5

# --- Freeze + Fine-tuning schedule ---
FREEZE_EPOCHS = 20  # 先训练 head
UNFREEZE_BACKBONE_LR = 1e-5  # 解冻后 backbone LR

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 25
EARLY_STOPPING_MIN_DELTA = 1e-4

# --- Validation ---
USE_TRUE_VAL_SET = True
VAL_RATIO = 0.2
VAL_SEED = 42

# --- Count Calibration ---
COUNT_TEMP = 1.0
USE_COUNT_THRESHOLD = False
COUNT_THRESHOLD = 0.5
LAMBDA_COUNT = 0.05

# --- Border / Out-of-bounds Penalties ---
USE_BORDER_PENALTY = True
BORDER_MARGIN = 16  # pixels near image border
BORDER_PENALTY_WEIGHT = 2.0
OOB_PENALTY_WEIGHT = 4.0
BORDER_PENALTY_WITH_SCORE = True

# --- Debug / Visualization ---
TRAIN_VIS_DIR = "results_train/vis_results"
TRAIN_VIS_EVERY = 200
TRAIN_VIS_PER_EPOCH = 3
TRAIN_VIS_SCORE_THR = 0.5
LOG_MATCH_STATS = True

# --- Matcher Costs ---
MATCH_COST_CLASS = 1.0
MATCH_COST_POINT = 1.3

def _denorm_to_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    img_np = img_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
    return img_np

def _filter_points_in_bounds(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return points
    mask = (
        (points[:, 0] >= 0) & (points[:, 0] < width) &
        (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    return points[mask]

def _count_points_out_of_bounds(points: np.ndarray, width: int, height: int) -> int:
    if points.size == 0:
        return 0
    mask = (
        (points[:, 0] < 0) | (points[:, 0] >= width) |
        (points[:, 1] < 0) | (points[:, 1] >= height)
    )
    return int(mask.sum())

def _save_train_vis(img_tensor: torch.Tensor,
                    gt_points: torch.Tensor,
                    pred_points: torch.Tensor,
                    pred_scores: torch.Tensor,
                    out_path: str,
                    score_thr: float,
                    border_margin: int):
    img_np = _denorm_to_uint8(img_tensor)
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    gt_np = gt_points.detach().cpu().numpy()
    gt_np = _filter_points_in_bounds(gt_np, w, h)
    for p in gt_np:
        cv2.circle(img_cv, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

    pred_np = pred_points.detach().cpu().numpy()
    score_np = pred_scores.detach().cpu().numpy()
    keep = score_np > score_thr
    pred_np = pred_np[keep]
    pred_np = _filter_points_in_bounds(pred_np, w, h)
    for p in pred_np:
        x, y = int(p[0]), int(p[1])
        if x < border_margin or x >= (w - border_margin) or y < border_margin or y >= (h - border_margin):
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(img_cv, (x, y), 2, color, -1)

    cv2.imwrite(out_path, img_cv)


def main():
    # 1. 检查目录
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    if TRAIN_VIS_EVERY > 0:
        os.makedirs(TRAIN_VIS_DIR, exist_ok=True)
        
    print(f"Running on {DEVICE}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please enable GPU or install CUDA-enabled PyTorch.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 2. 数据集
    def _build_part_dataset(part_name: str, split: str, method: str) -> CrowdDataset:
        return CrowdDataset(
            DATA_ROOT,
            part=part_name,
            split=split,
            method=method,
            jitter_strength=0.1,
            blur_prob=0.1,
            blur_radius=0.8,
        )

    base_train = ConcatDataset([
        _build_part_dataset('part_A_final', 'train', 'train'),
        _build_part_dataset('part_B_final', 'train', 'train'),
    ])

    if USE_TRUE_VAL_SET:
        train_dataset = base_train
        val_dataset = ConcatDataset([
            _build_part_dataset('part_A_final', 'test', 'test'),
            _build_part_dataset('part_B_final', 'test', 'test'),
        ])
    else:
        base_val = ConcatDataset([
            _build_part_dataset('part_A_final', 'train', 'test'),
            _build_part_dataset('part_B_final', 'train', 'test'),
        ])
        total_size = len(base_train)
        val_size = max(1, int(total_size * VAL_RATIO))
        train_size = total_size - val_size
        generator = torch.Generator().manual_seed(VAL_SEED)
        train_split, val_split = random_split(range(total_size), [train_size, val_size], generator=generator)

        train_dataset = Subset(base_train, train_split.indices)
        val_dataset = Subset(base_val, val_split.indices)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. 模型
    model = P2PNet().to(DEVICE)
    
    # 4. 损失函数 & 优化器
    matcher = HungarianMatcher(cost_class=MATCH_COST_CLASS, cost_point=MATCH_COST_POINT)
    criterion = P2PNetLoss(
        matcher,
        lambda_neg=LAMBDA_NEG,
        use_focal=USE_FOCAL_LOSS,
        focal_alpha=FOCAL_ALPHA,
        focal_gamma=FOCAL_GAMMA,
    ).to(DEVICE)

    # --- Freeze backbone for warmup ---
    for p in model.backbone.parameters():
        p.requires_grad = False

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

    # 5. 训练循环
    print("Start Training...")
    train_metrics = []
    best_val_mae = float("inf")
    epochs_no_improve = 0
    last_epoch = 0
    for epoch in range(EPOCHS):
        # --- Unfreeze backbone after FREEZE_EPOCHS ---
        if epoch == FREEZE_EPOCHS:
            print("Unfreezing backbone for fine-tuning...")
            for p in model.backbone.parameters():
                p.requires_grad = True

            param_dicts = [
                {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
                {
                    "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                    "lr": UNFREEZE_BACKBONE_LR,
                },
            ]
            optimizer = torch.optim.AdamW(param_dicts, lr=LR, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)

        model.train()
        epoch_loss = 0
        vis_saved = 0
        
        for i, (samples, targets) in enumerate(train_loader):
            samples = samples.to(DEVICE)
            # targets 需要把其中的 tensor 也移到 device
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            logits = outputs['pred_logits'].squeeze(-1)
            scores = torch.sigmoid(logits / COUNT_TEMP)
            if USE_COUNT_THRESHOLD:
                pred_counts = (scores > COUNT_THRESHOLD).sum(dim=1)
            else:
                pred_counts = scores.sum(dim=1)
            gt_counts = torch.tensor([len(t['point']) for t in targets], device=pred_counts.device, dtype=pred_counts.dtype)
            count_loss = F.l1_loss(pred_counts, gt_counts, reduction='mean')
            losses = losses + (LAMBDA_COUNT * count_loss)

            # --- Border / OOB penalties ---
            if USE_BORDER_PENALTY:
                pred_points = outputs['pred_points']  # [B, Q, 2]
                h_img, w_img = samples.shape[2], samples.shape[3]

                x = pred_points[..., 0]
                y = pred_points[..., 1]

                # Out-of-bounds penalty
                oob = (
                    F.relu(-x) +
                    F.relu(x - (w_img - 1)) +
                    F.relu(-y) +
                    F.relu(y - (h_img - 1))
                )

                # Border margin penalty (inside image, near edges)
                dist_left = x
                dist_right = (w_img - 1) - x
                dist_top = y
                dist_bottom = (h_img - 1) - y
                dist_edge = torch.stack([dist_left, dist_right, dist_top, dist_bottom], dim=-1).min(dim=-1).values
                border_violation = F.relu(BORDER_MARGIN - dist_edge) / max(BORDER_MARGIN, 1)

                if BORDER_PENALTY_WITH_SCORE:
                    weight = scores
                    oob_pen = (oob * weight).mean()
                    border_pen = (border_violation * weight).mean()
                else:
                    oob_pen = oob.mean()
                    border_pen = border_violation.mean()

                losses = losses + (OOB_PENALTY_WEIGHT * oob_pen) + (BORDER_PENALTY_WEIGHT * border_pen)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
            if i % 10 == 0:
                if USE_BORDER_PENALTY:
                    with torch.no_grad():
                        oob_rate = (oob > 0).float().mean().item()
                        border_rate = (border_violation > 0).float().mean().item()
                        conf_mask = scores > TRAIN_VIS_SCORE_THR
                        if conf_mask.any():
                            oob_conf = (oob[conf_mask] > 0).float().mean().item()
                            border_conf = (border_violation[conf_mask] > 0).float().mean().item()
                        else:
                            oob_conf = 0.0
                            border_conf = 0.0
                    print(
                        f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f} | "
                        f"OOB rate: {oob_rate:.4f}, Border rate: {border_rate:.4f} | "
                        f"OOB conf: {oob_conf:.4f}, Border conf: {border_conf:.4f}"
                    )
                else:
                    print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f}")

            # --- Debug visualization + matcher stats ---
            if TRAIN_VIS_EVERY > 0 and (i % TRAIN_VIS_EVERY == 0) and (vis_saved < TRAIN_VIS_PER_EPOCH):
                with torch.no_grad():
                    if LOG_MATCH_STATS:
                        indices = matcher(outputs, targets)
                        matched = sum(len(src) for src, _ in indices)
                        total_gt = sum(len(t['point']) for t in targets)
                        print(f"[Match] Epoch {epoch} Step {i}: matched={matched} / gt={total_gt}")

                    img_tensor = samples[0]
                    gt_points = targets[0]['point']
                    pred_points = outputs['pred_points'][0]
                    pred_scores = scores[0]
                    out_path = os.path.join(TRAIN_VIS_DIR, f"epoch{epoch}_step{i}_b0.jpg")
                    _save_train_vis(
                        img_tensor,
                        gt_points,
                        pred_points,
                        pred_scores,
                        out_path,
                        TRAIN_VIS_SCORE_THR,
                        BORDER_MARGIN,
                    )
                    vis_saved += 1

        # --- Validation ---
        model.eval()
        val_mae = 0.0
        val_mse = 0.0
        val_count = 0

        with torch.no_grad():
            for samples, targets in val_loader:
                samples = samples.to(DEVICE)
                outputs = model(samples)

                logits = outputs['pred_logits'].squeeze(-1)
                scores = torch.sigmoid(logits / COUNT_TEMP)
                if USE_COUNT_THRESHOLD:
                    pred_counts = (scores > COUNT_THRESHOLD).sum(dim=1)
                else:
                    pred_counts = scores.sum(dim=1)

                for b in range(len(targets)):
                    gt_count = len(targets[b]['point'])
                    pred_count = pred_counts[b].item()
                    val_mae += abs(pred_count - gt_count)
                    val_mse += (pred_count - gt_count) ** 2
                    val_count += 1

        val_mae = val_mae / max(1, val_count)
        val_rmse = (val_mse / max(1, val_count)) ** 0.5

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        avg_loss = epoch_loss / len(train_loader)
        train_metrics.append((epoch + 1, avg_loss, current_lr, val_mae, val_rmse))
        print(
            f"--- Epoch {epoch} Finished. Avg Loss: {avg_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | LR: {current_lr:.6f} ---"
        )

        last_epoch = epoch + 1

        # Early stopping on validation MAE
        if val_mae < (best_val_mae - EARLY_STOPPING_MIN_DELTA):
            best_val_mae = val_mae
            epochs_no_improve = 0
            best_path = os.path.join(WEIGHT_DIR, "best.pth")
            torch.save({'model': model.state_dict()}, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch} (no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
                break
        
        # 保存权重
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(WEIGHT_DIR, f"epoch_{epoch+1}.pth")
            torch.save({'model': model.state_dict()}, ckpt_path)

        if (epoch + 1) == EPOCHS:
            latest_path = os.path.join(WEIGHT_DIR, "latest.pth")
            torch.save({'model': model.state_dict()}, latest_path)

    # Save latest checkpoint (even if early-stopped)
    if last_epoch > 0:
        latest_path = os.path.join(WEIGHT_DIR, "latest.pth")
        torch.save({'model': model.state_dict()}, latest_path)

    # 保存训练结果 (CSV + 图)
    csv_path = os.path.join(RESULTS_DIR, "train_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "avg_loss", "lr", "val_mae", "val_rmse"])
        writer.writerows(train_metrics)

    epochs = [m[0] for m in train_metrics]
    losses = [m[1] for m in train_metrics]
    lrs = [m[2] for m in train_metrics]
    val_maes = [m[3] for m in train_metrics]
    val_rmses = [m[4] for m in train_metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, losses, marker='o', label='Train Loss')
    plt.title('Training Loss vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "train_loss.png")
    plt.savefig(plot_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_maes, marker='o', label='Val MAE')
    plt.plot(epochs, val_rmses, marker='o', label='Val RMSE')
    plt.title('Validation Metrics vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    val_plot_path = os.path.join(RESULTS_DIR, "val_metrics.png")
    plt.savefig(val_plot_path)
    plt.close()

    # Summary text
    summary_path = os.path.join(RESULTS_DIR, "train_summary.txt")
    best_loss = min(train_metrics, key=lambda x: x[1])
    best_val = min(train_metrics, key=lambda x: x[3])
    with open(summary_path, "w") as f:
        f.write(f"Best Avg Loss: epoch {best_loss[0]} -> {best_loss[1]:.6f}\n")
        f.write(f"Best Val MAE: epoch {best_val[0]} -> {best_val[3]:.6f}\n")
        f.write(f"Final LR: {lrs[-1]:.8f}\n")
        f.write(f"Last Epoch: {last_epoch}\n")

if __name__ == "__main__":
    main()