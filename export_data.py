import os
import cv2
import numpy as np
import scipy.io as io
import math
import glob
import random
from PIL import Image, ImageEnhance, ImageFilter

# Use tqdm if installed, otherwise pass through
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

# Paths
INPUT_ROOT = "Dataset/ShanghaiTech"
OUTPUT_ROOT = "expd/ShanghaiTech"
TARGET_SIZE = 384

# Augmentations (applied only to train_data when ENABLE_AUGMENT=True)
ENABLE_AUGMENT = True
JITTER_STRENGTH = 0.2
BLUR_PROB = 0.3
BLUR_RADIUS = 1.2

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
            print(f"[{part}/{split}] Found {len(img_paths)} images. Starting preprocessing...")

            for img_path in tqdm(img_paths):
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue

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

                # 1. Resize keep-aspect (short side to TARGET_SIZE)
                h, w = img.shape[:2]
                scale = TARGET_SIZE / min(h, w)
                new_h = int(round(h * scale))
                new_w = int(round(w * scale))
                if new_h <= 0 or new_w <= 0:
                    continue

                img = cv2.resize(img, (new_w, new_h))
                points *= scale

                # 2. Augmentations (train split only)
                if ENABLE_AUGMENT and split == "train_data":
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(img_rgb)

                    if random.random() < 0.8:
                        factor = 1.0 + random.uniform(-JITTER_STRENGTH, JITTER_STRENGTH)
                        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)
                    if random.random() < 0.8:
                        factor = 1.0 + random.uniform(-JITTER_STRENGTH, JITTER_STRENGTH)
                        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)
                    if random.random() < 0.8:
                        factor = 1.0 + random.uniform(-JITTER_STRENGTH, JITTER_STRENGTH)
                        pil_img = ImageEnhance.Color(pil_img).enhance(factor)
                    if BLUR_PROB > 0 and random.random() < BLUR_PROB:
                        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))

                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

                # Save processed data
                cv2.imwrite(os.path.join(out_imgs_dir, fname), img)
                np.save(os.path.join(out_gts_dir, fname.replace('.jpg', '.npy')), points)

            total_imgs += len(img_paths)

    print(f"Preprocessing completed! Data saved to {OUTPUT_ROOT}. Total images: {total_imgs}")

if __name__ == "__main__":
    preprocess_dataset()