import torch
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
from dataset import CrowdDataset, collate_fn
from p2pnet import P2PNet, HungarianMatcher
import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt

# --- Config ---
DATA_ROOT = "expd/ShanghaiTech"
WEIGHT_PATH = "weights/epoch_30.pth"
OUTPUT_DIR = "vis_results"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_VIS_COUNT = 3
MATCH_DIST_THRESH = 8.0

def _list_weight_paths(weights_dir: str) -> list[str]:
    paths = []
    for name in os.listdir(weights_dir):
        match = re.match(r"epoch_(\d+)\.pth$", name)
        if not match:
            continue
        path = os.path.join(weights_dir, name)
        if os.path.getsize(path) == 0:
            continue
        paths.append(path)
    paths.sort(key=lambda p: int(re.search(r"epoch_(\d+)", p).group(1)))
    return paths


def _list_special_weights(weights_dir: str) -> list[str]:
    specials = []
    for name in ["best.pth", "latest.pth"]:
        path = os.path.join(weights_dir, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            specials.append(path)
    return specials


def evaluate_and_visualize(weight_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset (Method='test' 保持原图)
    # Batch Size 必须为 1，因为每张原图大小不一样
    test_set = ConcatDataset([
        CrowdDataset(DATA_ROOT, part='part_A_final', split='test', method='test'),
        CrowdDataset(DATA_ROOT, part='part_B_final', split='test', method='test'),
    ])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 2. Model
    model = P2PNet().to(DEVICE)
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded weights from {weight_path}")
    else:
        print("Weights not found!")
        return
        
    model.eval()
    
    mae = 0
    mse = 0
    total_imgs = len(test_loader)
    matcher = HungarianMatcher(cost_class=1.0, cost_point=1.0)
    matched_dists = []
    matched_within = 0
    matched_total = 0
    
    print(f"Starting evaluation on {total_imgs} images...")
    
    saved_vis = 0

    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(DEVICE)
            gt_count = len(targets[0]['point'])
            
            outputs = model(imgs)
            indices = matcher(outputs, targets)
            for b, (src_idx, tgt_idx) in enumerate(indices):
                if src_idx.numel() == 0:
                    continue
                pred_pts = outputs['pred_points'][b, src_idx]
                tgt_pts = targets[b]['point'][tgt_idx].to(pred_pts.device)
                dists = torch.norm(pred_pts - tgt_pts, dim=1)
                matched_dists.append(dists.detach().cpu())
                matched_within += int((dists <= MATCH_DIST_THRESH).sum().item())
                matched_total += int(dists.numel())
            
            # --- Counting Metrics (Tutorial Section) ---
            logits = outputs['pred_logits'].view(-1)
            points = outputs['pred_points'].view(-1, 2)
            scores = torch.sigmoid(logits)
            
            # 1. Hard Count (Threshold = 0.5)
            mask = scores > 0.5
            pred_count_hard = mask.sum().item()
            
            # 2. Soft Count (Sum of scores)
            pred_count_soft = scores.sum().item()
            
            # 使用 Soft Count 计算 MAE/MSE (通常更准，也可以换成 Hard)
            pred_count = pred_count_soft 
            
            mae += abs(pred_count - gt_count)
            mse += (pred_count - gt_count) ** 2
            
            if i % 20 == 0:
                print(f"Img {i}: GT={gt_count}, Pred={pred_count:.2f} (Soft), {pred_count_hard} (Hard)")

            # --- Visualization (Save multiple images) ---
            if saved_vis < SAVE_VIS_COUNT:
                img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).astype(np.uint8)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Draw GT (Green)
                for p in targets[0]['point']:
                    cv2.circle(img_cv, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

                # Draw Pred (Red) - Only confident ones
                valid_points = points[mask].cpu().numpy()
                for p in valid_points:
                    cv2.circle(img_cv, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

                cv2.imwrite(f"{OUTPUT_DIR}/res_{i}_gt{gt_count}_pred{int(pred_count)}.jpg", img_cv)
                saved_vis += 1

    mae = mae / total_imgs
    mse = (mse / total_imgs) ** 0.5 # RMSE

    if matched_dists:
        all_dists = torch.cat(matched_dists, dim=0)
        mean_dist = all_dists.mean().item()
        median_dist = all_dists.median().item()
        acc_dist = matched_within / max(1, matched_total)
        print(f"MatchDist mean/median: {mean_dist:.2f}/{median_dist:.2f} px | Acc@{MATCH_DIST_THRESH:.0f}px={acc_dist:.3f}")
    
    print("=========================================")
    print(f"Final Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}") # 实际上是 RMSE
    print("=========================================")

    return mae, mse


def evaluate_all_checkpoints():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Dataset (Method='test' 保持原图)
    test_set = ConcatDataset([
        CrowdDataset(DATA_ROOT, part='part_A_final', split='test', method='test'),
        CrowdDataset(DATA_ROOT, part='part_B_final', split='test', method='test'),
    ])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    weights_dir = os.path.dirname(WEIGHT_PATH) or "weights"
    weight_paths = _list_weight_paths(weights_dir)
    special_paths = _list_special_weights(weights_dir)
    all_paths = weight_paths + special_paths

    if not all_paths:
        print("No valid weight files found!")
        return

    metrics = []
    matcher = HungarianMatcher(cost_class=1.0, cost_point=1.0)

    for weight_path in all_paths:
        model = P2PNet().to(DEVICE)
        checkpoint = torch.load(weight_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        mae = 0
        mse = 0
        total_imgs = len(test_loader)
        matched_dists = []
        matched_within = 0
        matched_total = 0
        saved_vis = 0

        with torch.no_grad():
            for i, (imgs, targets) in enumerate(test_loader):
                imgs = imgs.to(DEVICE)
                gt_count = len(targets[0]['point'])

                outputs = model(imgs)
                indices = matcher(outputs, targets)
                for b, (src_idx, tgt_idx) in enumerate(indices):
                    if src_idx.numel() == 0:
                        continue
                    pred_pts = outputs['pred_points'][b, src_idx]
                    tgt_pts = targets[b]['point'][tgt_idx].to(pred_pts.device)
                    dists = torch.norm(pred_pts - tgt_pts, dim=1)
                    matched_dists.append(dists.detach().cpu())
                    matched_within += int((dists <= MATCH_DIST_THRESH).sum().item())
                    matched_total += int(dists.numel())
                logits = outputs['pred_logits'].view(-1)
                points = outputs['pred_points'].view(-1, 2)
                scores = torch.sigmoid(logits)

                pred_count_soft = scores.sum().item()
                pred_count = pred_count_soft

                mae += abs(pred_count - gt_count)
                mse += (pred_count - gt_count) ** 2

                # Visualization: Save 2 images per epoch
                if saved_vis < 2:
                    mask = scores > 0.45

                    img_np = imgs[0].cpu().permute(1, 2, 0).numpy()
                    img_np = (img_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                    img_np = (img_np * 255).astype(np.uint8)
                    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                    # Draw GT (Green)
                    for p in targets[0]['point']:
                        cv2.circle(img_cv, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)

                    # Draw Pred (Red) - Only confident ones
                    valid_points = points[mask].cpu().numpy()
                    for p in valid_points:
                        cv2.circle(img_cv, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

                    epoch_match = re.search(r"epoch_(\d+)", weight_path)
                    epoch_num = int(epoch_match.group(1)) if epoch_match else -1
                    cv2.imwrite(f"{OUTPUT_DIR}/epoch{epoch_num}_img{i}_gt{gt_count}_pred{int(pred_count)}.jpg", img_cv)
                    saved_vis += 1

        mae = mae / total_imgs
        mse = (mse / total_imgs) ** 0.5

        epoch_match = re.search(r"epoch_(\d+)", weight_path)
        if matched_dists:
            all_dists = torch.cat(matched_dists, dim=0)
            mean_dist = all_dists.mean().item()
            median_dist = all_dists.median().item()
            acc_dist = matched_within / max(1, matched_total)
            print(f"MatchDist mean/median: {mean_dist:.2f}/{median_dist:.2f} px | Acc@{MATCH_DIST_THRESH:.0f}px={acc_dist:.3f}")

        if epoch_match:
            label = f"epoch_{int(epoch_match.group(1))}"
            print(f"Epoch {int(epoch_match.group(1))}: MAE={mae:.4f}, MSE={mse:.4f}")
        else:
            label = os.path.splitext(os.path.basename(weight_path))[0]
            print(f"{label}: MAE={mae:.4f}, MSE={mse:.4f}")

        metrics.append((label, mae, mse))

    epoch_metrics = [m for m in metrics if m[0].startswith("epoch_")]
    epoch_metrics.sort(key=lambda x: int(x[0].split("_")[1]))

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint", "mae", "rmse"])
        writer.writerows(metrics)

    # Plot
    epochs = [int(m[0].split("_")[1]) for m in epoch_metrics]
    maes = [m[1] for m in epoch_metrics]
    rmses = [m[2] for m in epoch_metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, maes, marker='o', label='MAE')
    plt.plot(epochs, rmses, marker='o', label='RMSE')
    plt.title('P2PNet Performance vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "metrics.png")
    plt.savefig(plot_path)
    plt.close()

    # Summary text
    summary_path = os.path.join(RESULTS_DIR, "summary.txt")
    best_mae = min(metrics, key=lambda x: x[1])
    best_rmse = min(metrics, key=lambda x: x[2])
    with open(summary_path, "w") as f:
        f.write(f"Best MAE: {best_mae[0]} -> {best_mae[1]:.4f}\n")
        f.write(f"Best RMSE: {best_rmse[0]} -> {best_rmse[2]:.4f}\n")

    print(f"Saved results to {RESULTS_DIR}")

if __name__ == "__main__":
    # Evaluate all saved epochs and generate metrics + plots
    evaluate_all_checkpoints()