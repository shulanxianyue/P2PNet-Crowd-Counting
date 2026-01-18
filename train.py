import torch
import os
import time
import csv
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from dataset import CrowdDataset, collate_fn
from p2pnet import P2PNet, P2PNetLoss, HungarianMatcher

# --- Config ---
DATA_ROOT = "expd/ShanghaiTech"
EPOCHS = 200
BATCH_SIZE = 2  # 如果显存不够(报错OOM)，把这里改成 2 或 1
VAL_BATCH_SIZE = 1
LR = 1e-4
LR_BACKBONE = 1e-4
WEIGHT_DIR = "weights"
RESULTS_DIR = "results_train"
DEVICE = "cuda"  # <--- 关键补充

# Loss weight for negatives
LAMBDA_NEG = 4.0

# --- Focal Loss ---
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 3.0

# --- Scheduler ---
SCHEDULER_STEP = 40
SCHEDULER_GAMMA = 0.5

# --- Freeze + Fine-tuning schedule ---
FREEZE_EPOCHS = 20  # 先训练 head
UNFREEZE_BACKBONE_LR = 1e-5  # 解冻后 backbone LR

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 1e-4

# --- Validation ---
USE_TRUE_VAL_SET = True
VAL_RATIO = 0.2
VAL_SEED = 42

# --- Count Calibration ---
COUNT_TEMP = 1.0
USE_COUNT_THRESHOLD = False
COUNT_THRESHOLD = 0.5
LAMBDA_COUNT = 0.1

def main():
    # 1. 检查目录
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    print(f"Running on {DEVICE}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Please enable GPU or install CUDA-enabled PyTorch.")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # 2. 数据集
    base_train = CrowdDataset(DATA_ROOT, part='part_A_final', split='train', method='train')

    if USE_TRUE_VAL_SET:
        train_dataset = base_train
        val_dataset = CrowdDataset(DATA_ROOT, part='part_A_final', split='test', method='test')
    else:
        base_val = CrowdDataset(DATA_ROOT, part='part_A_final', split='train', method='test')
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
    matcher = HungarianMatcher()
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

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f}")

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