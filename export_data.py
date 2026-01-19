import os
import cv2
import numpy as np
import scipy.io as io
import glob

# Use tqdm if installed, otherwise pass through
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# Paths
INPUT_ROOT = ".data/ShanghaiTech"
OUTPUT_ROOT = "expd/ShanghaiTech"

PARTS = ["part_A_final", "part_B_final"]
SPLITS = ["train_data", "test_data"]

def preprocess_dataset():
    total_imgs = 0

    for part in PARTS:
        for split in SPLITS:
            in_imgs_dir = os.path.join(INPUT_ROOT, part, split, "images")
            in_gts_dir = os.path.join(INPUT_ROOT, part, split, "ground_truth")

            out_imgs_dir = os.path.join(OUTPUT_ROOT, part, split, "images")
            out_gts_dir = os.path.join(OUTPUT_ROOT, part, split, "ground_truth")

            os.makedirs(out_imgs_dir, exist_ok=True)
            os.makedirs(out_gts_dir, exist_ok=True)

            img_paths = glob.glob(os.path.join(in_imgs_dir, "*.jpg"))
            print(f"[{part}/{split}] Found {len(img_paths)} images. Starting export...")

            for img_path in tqdm(img_paths):
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Load corresponding mat file
                fname = os.path.basename(img_path)
                gt_name = "GT_" + fname.replace('.jpg', '.mat')
                gt_path = os.path.join(in_gts_dir, gt_name)

                points = []
                if os.path.exists(gt_path):
                    try:
                        # Parse ShanghaiTech mat format
                        mat = io.loadmat(gt_path)
                        points = mat['image_info'][0, 0][0, 0][0]
                    except Exception:
                        pass

                points = np.array(points, dtype=np.float32).reshape(-1, 2)
                if points.size > 0:
                    mask = (points[:, 0] >= 0) & (points[:, 0] < w) & \
                           (points[:, 1] >= 0) & (points[:, 1] < h)
                    points = points[mask]

                # Save original image and exported GT points (.npy)
                cv2.imwrite(os.path.join(out_imgs_dir, fname), img)
                np.save(os.path.join(out_gts_dir, fname.replace('.jpg', '.npy')), points)

            total_imgs += len(img_paths)

    print(f"Preprocessing completed! Data saved to {OUTPUT_ROOT}. Total images: {total_imgs}")

if __name__ == "__main__":
    preprocess_dataset()