import torch
import numpy as np
import os
import random
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageEnhance, ImageFilter
from scipy.io import loadmat

class CrowdDataset(Dataset):
    def __init__(
        self,
        root,
        part='part_A_final',
        split='train',
        crop_size=384,
        method='train',
        validate_gt=True,
        gt_check_samples=5,
        use_augment=True,
        hflip_prob=0.5,
        jitter_strength=0.2,
        blur_prob=0.3,
        blur_radius=1.2,
    ):
        """
        method: 'train' (随机裁剪), 'val' (原图), 'test' (原图)
        """
        self.root = root
        self.split = split
        self.method = method
        self.crop_size = crop_size
        self.use_augment = use_augment
        self.hflip_prob = hflip_prob
        self.jitter_strength = jitter_strength
        self.blur_prob = blur_prob
        self.blur_radius = blur_radius
        
        # 1. 确定图片和GT的子文件夹
        if split == 'train':
            sub_dir = 'train_data'
        else:
            sub_dir = 'test_data'
            
        # 拼凑路径 (支持原始 Dataset 和预处理后的 expd)
        path_img = os.path.join(root, part, sub_dir, 'images')
        path_gt = os.path.join(root, part, sub_dir, 'ground_truth')

        # 2. 获取所有图片文件名
        # 只读取文件名，不包含路径
        if not os.path.exists(path_img):
            raise FileNotFoundError(f"Image path not found: {path_img}")
        if not os.path.exists(path_gt):
            raise FileNotFoundError(f"GT path not found: {path_gt}")

        img_names = [f for f in os.listdir(path_img) if f.endswith('.jpg')]

        # 判断当前数据使用 .npy 还是 .mat 标注
        gt_files = os.listdir(path_gt)
        has_npy = any(f.endswith('.npy') for f in gt_files)
        has_mat = any(f.endswith('.mat') for f in gt_files)
        if has_npy:
            self.gt_ext = '.npy'
            self.gt_prefix = ''
        elif has_mat:
            self.gt_ext = '.mat'
            self.gt_prefix = 'GT_'
        else:
            raise RuntimeError(f"No supported annotation files found in {path_gt}")

        if has_npy and has_mat:
            print(f"[WARN] Both .npy and .mat found in {path_gt}. Using .npy. Ensure images match the .npy pipeline.")
        
        self.img_paths = []
        self.gt_paths = []

        # 3. 构造完整路径 & 匹配 .npy 文件
        for img_name in img_names:
            # 图片完整路径
            self.img_paths.append(os.path.join(path_img, img_name))
            
            if self.gt_ext == '.npy':
                gt_name = img_name.replace('.jpg', '.npy')
            else:
                base = img_name.replace('.jpg', '.mat')
                if not base.startswith('GT_'):
                    base = f"{self.gt_prefix}{base}"
                gt_name = base
            
            # GT 完整路径
            self.gt_paths.append(os.path.join(path_gt, gt_name))
            
        print(f"[{split.upper()}] Loaded {len(self.img_paths)} images from {path_img}")

        if validate_gt:
            self._validate_gt_alignment(sample_size=gt_check_samples)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1. 加载图片和点
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        
        # 加载图片
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        points = self._load_points(gt_path)

        # 2. 预处理 (Train vs Test)
        if self.method == 'train':
            # 保持长宽比：短边缩放到 crop_size
            img, points = self._resize_keep_aspect(img, points, self.crop_size)
            w, h = img.size

            # --- 训练：随机裁剪固定大小 (384x384) ---
            i, j, th, tw = self._random_crop_params(h, w, self.crop_size, self.crop_size)

            img = img.crop((j, i, j + tw, i + th))

            if len(points) > 0:
                points[:, 0] -= j
                points[:, 1] -= i
                mask = (points[:, 0] >= 0) & (points[:, 0] < tw) & \
                       (points[:, 1] >= 0) & (points[:, 1] < th)
                points = points[mask]

            # --- Data Augmentations ---
            if self.use_augment:
                img, points = self._random_hflip(img, points)
                img = self._color_jitter(img)
        
        else:
            # --- 测试/验证：保持长宽比，仅缩放短边到 crop_size（不裁剪） ---
            img, points = self._resize_keep_aspect(img, points, self.crop_size)
            w, h = img.size

        # 3. 归一化 (Standard ImageNet mean/std)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        points_tensor = torch.tensor(points, dtype=torch.float32)

        return img_tensor, {'point': points_tensor}

    def _random_crop_params(self, h, w, th, tw):
        if w < tw and h < th:
            # 如果两个方向都比裁剪尺寸小，就不裁切
            return 0, 0, h, w

        if h < th:
            th = h
        if w < tw:
            tw = w
        
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

    def _load_points(self, gt_path):
        try:
            if gt_path.endswith('.npy'):
                points = np.load(gt_path).astype(np.float32)
            else:
                mat = loadmat(gt_path)
                points = mat['image_info'][0, 0][0, 0][0].astype(np.float32)
        except Exception as e:
            print(f"Error loading {gt_path}: {e}")
            points = np.zeros((0, 2), dtype=np.float32)

        return points.reshape(-1, 2)

    def _validate_gt_alignment(self, sample_size=5, tol=1.0):
        if len(self.img_paths) == 0:
            return

        sample_size = min(sample_size, len(self.img_paths))
        bad = 0

        for idx in range(sample_size):
            img_path = self.img_paths[idx]
            gt_path = self.gt_paths[idx]

            try:
                img = Image.open(img_path).convert('RGB')
                w, h = img.size
                points = self._load_points(gt_path)
            except Exception as e:
                print(f"[WARN] GT alignment check failed for {img_path}: {e}")
                bad += 1
                continue

            if len(points) == 0:
                continue

            out_x = (points[:, 0] < -tol) | (points[:, 0] > (w - 1 + tol))
            out_y = (points[:, 1] < -tol) | (points[:, 1] > (h - 1 + tol))
            if (out_x | out_y).any():
                bad += 1

        if bad > 0:
            print(
                "[WARN] GT points appear out of image bounds for some samples. "
                "Check that images and annotations come from the same pipeline (original .mat with original images, "
                "or preprocessed .npy with preprocessed images)."
            )

    def _resize_keep_aspect(self, img, points, target):
        w, h = img.size
        if w == 0 or h == 0:
            return img, points

        scale = target / min(w, h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        if new_w == w and new_h == h:
            return img, points

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if len(points) > 0:
            points = points.copy()
            points[:, 0] *= new_w / w
            points[:, 1] *= new_h / h

        return img, points

    def _random_hflip(self, img, points):
        if random.random() < self.hflip_prob:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if len(points) > 0:
                w = img.size[0]
                points = points.copy()
                points[:, 0] = (w - 1) - points[:, 0]
        return img, points

    def _color_jitter(self, img):
        if self.jitter_strength <= 0:
            return img

        # Brightness
        if random.random() < 0.8:
            factor = 1.0 + random.uniform(-self.jitter_strength, self.jitter_strength)
            img = ImageEnhance.Brightness(img).enhance(factor)

        # Contrast
        if random.random() < 0.8:
            factor = 1.0 + random.uniform(-self.jitter_strength, self.jitter_strength)
            img = ImageEnhance.Contrast(img).enhance(factor)

        # Saturation
        if random.random() < 0.8:
            factor = 1.0 + random.uniform(-self.jitter_strength, self.jitter_strength)
            img = ImageEnhance.Color(img).enhance(factor)

        # Background blur
        if self.blur_prob > 0 and random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        return img


def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]

    # Pad to max size in batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    batch_size = len(images)

    padded = images[0].new_zeros((batch_size, 3, max_h, max_w))
    for i, img in enumerate(images):
        c, h, w = img.shape
        padded[i, :, :h, :w] = img

    return padded, targets