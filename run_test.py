import torch
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader
from dataset import CrowdDataset, collate_fn
from p2pnet import P2PNet

# --- Config ---
DATA_ROOT = "Dataset/ShanghaiTech"
WEIGHT_PATH = "weights/latest.pth"
OUTPUT_DIR = "vis_results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate_and_visualize():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Dataset (Method='test' 保持原图)
    # Batch Size 必须为 1，因为每张原图大小不一样
    test_set = CrowdDataset(DATA_ROOT, split='test', method='test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 2. Model
    model = P2PNet().to(DEVICE)
    if os.path.exists(WEIGHT_PATH):
        checkpoint = torch.load(WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded weights from {WEIGHT_PATH}")
    else:
        print("Weights not found!")
        return
        
    model.eval()
    
    mae = 0
    mse = 0
    total_imgs = len(test_loader)
    
    print(f"Starting evaluation on {total_imgs} images...")
    
    with torch.no_grad():
        for i, (imgs, targets) in enumerate(test_loader):
            imgs = imgs.to(DEVICE)
            gt_count = len(targets[0]['point'])
            
            outputs = model(imgs)
            
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
                
                # --- Visualization (Save first few images) ---
                if i < 5:
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

    mae = mae / total_imgs
    mse = (mse / total_imgs) ** 0.5 # RMSE
    
    print("=========================================")
    print(f"Final Results:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}") # 实际上是 RMSE
    print("=========================================")

if __name__ == "__main__":
    evaluate_and_visualize()