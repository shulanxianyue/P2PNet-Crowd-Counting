import torch
import os
import time
from torch.utils.data import DataLoader
from dataset import CrowdDataset, collate_fn
from p2pnet import P2PNet, P2PNetLoss, HungarianMatcher

# --- Config ---
DATA_ROOT = "Dataset/ShanghaiTech"
EPOCHS = 200
BATCH_SIZE = 4  # 如果显存不够(报错OOM)，把这里改成 2 或 1
LR = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DIR = "weights"
DEVICE = "cuda"  # <--- 关键补充

def main():
    # 1. 检查目录
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
        
    print(f"Running on {DEVICE}")

    # 2. 数据集
    train_dataset = CrowdDataset(DATA_ROOT, part='part_A_final', split='train', method='train')
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3. 模型
    model = P2PNet().to(DEVICE)
    
    # 4. 损失函数 & 优化器
    matcher = HungarianMatcher()
    criterion = P2PNetLoss(matcher).to(DEVICE)
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": LR_BACKBONE,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=LR, weight_decay=1e-4)

    # 5. 训练循环
    print("Start Training...")
    for epoch in range(EPOCHS):
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

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            epoch_loss += losses.item()
            
            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, Loss: {losses.item():.4f}")

        print(f"--- Epoch {epoch} Finished. Avg Loss: {epoch_loss / len(train_loader):.4f} ---")
        
        # 保存权重
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(WEIGHT_DIR, f"epoch_{epoch+1}.pth")
            torch.save({'model': model.state_dict()}, ckpt_path)

        if (epoch + 1) == EPOCHS:
            latest_path = os.path.join(WEIGHT_DIR, "latest.pth")
            torch.save({'model': model.state_dict()}, latest_path)

if __name__ == "__main__":
    main()