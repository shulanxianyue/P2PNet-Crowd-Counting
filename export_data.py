import os
import cv2
import numpy as np
import scipy.io as io
import math
import glob

# Use tqdm if installed, otherwise pass through
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# Paths
INPUT_ROOT = "Dataset/ShanghaiTech/part_A_final/train_data"
OUTPUT_ROOT = "expd/part_A_final/train_data"
TARGET_SIZE = 224

def preprocess_dataset():
    # Setup output directories
    out_imgs_dir = os.path.join(OUTPUT_ROOT, "images")
    out_gts_dir = os.path.join(OUTPUT_ROOT, "ground_truth")
    os.makedirs(out_imgs_dir, exist_ok=True)
    os.makedirs(out_gts_dir, exist_ok=True)

    # Load image list
    img_paths = glob.glob(os.path.join(INPUT_ROOT, "images", "*.jpg"))
    print(f"Found {len(img_paths)} images. Starting preprocessing...")

    for img_path in tqdm(img_paths):
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Load corresponding mat file
        fname = os.path.basename(img_path)
        gt_name = "GT_" + fname.replace('.jpg', '.mat')
        gt_path = os.path.join(INPUT_ROOT, "ground_truth", gt_name)
        
        points = []
        if os.path.exists(gt_path):
            try:
                # Parse ShanghaiTech mat format
                mat = io.loadmat(gt_path)
                points = mat['image_info'][0, 0][0, 0][0]
            except Exception:
                pass
        
        points = np.array(points, dtype=np.float32).reshape(-1, 2)

        # 1. Resize logic
        h, w = img.shape[:2]
        scale = TARGET_SIZE / min(h, w)
        new_h = math.ceil(h * scale)
        new_w = math.ceil(w * scale)
        
        # Ensure dimensions are at least target size
        new_h = max(new_h, TARGET_SIZE)
        new_w = max(new_w, TARGET_SIZE)
        
        img = cv2.resize(img, (new_w, new_h))
        points *= scale
        
        # 2. Center Crop logic
        dx = (new_w - TARGET_SIZE) // 2
        dy = (new_h - TARGET_SIZE) // 2
        
        img = img[dy:dy+TARGET_SIZE, dx:dx+TARGET_SIZE]
        
        points[:, 0] -= dx
        points[:, 1] -= dy
        
        # 3. Filter points outside the crop
        mask = (points[:, 0] >= 0) & (points[:, 0] < TARGET_SIZE) & \
               (points[:, 1] >= 0) & (points[:, 1] < TARGET_SIZE)
        points = points[mask]

        # Save processed data
        cv2.imwrite(os.path.join(out_imgs_dir, fname), img)
        np.save(os.path.join(out_gts_dir, fname.replace('.jpg', '.npy')), points)

    print(f"Preprocessing completed!  Data saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    preprocess_dataset()