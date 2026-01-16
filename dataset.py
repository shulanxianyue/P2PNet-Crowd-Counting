import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat

class CrowdDataset(Dataset):
    def __init__(self, root, part='part_A_final', split='train', crop_size=224, method='train'):
        """
        method: 'train' (随机裁剪), 'val' (原图), 'test' (原图)
        """
        self.root = root
        self.split = split
        self.method = method
        self.crop_size = crop_size
        
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
            img, points = self._ensure_min_size(img, points)
            w, h = img.size

            # --- 训练：随机裁剪固定大小 (224x224) ---
            i, j, th, tw = self._random_crop_params(h, w, self.crop_size, self.crop_size)

            img = img.crop((j, i, j + tw, i + th))

            if len(points) > 0:
                points[:, 0] -= j
                points[:, 1] -= i
                mask = (points[:, 0] >= 0) & (points[:, 0] < tw) & \
                       (points[:, 1] >= 0) & (points[:, 1] < th)
                points = points[mask]
        
        else:
            # --- 测试/验证：调整为 128 倍数 (Lanczos Resizing) ---
            img, points = self._resize_for_eval(img, points)

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

    def _ensure_min_size(self, img, points):
        w, h = img.size
        min_side = min(w, h)
        if min_side >= self.crop_size:
            return img, points

        scale = self.crop_size / max(min_side, 1)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

        if len(points) > 0:
            points = points.copy()
            points[:, 0] *= new_w / w
            points[:, 1] *= new_h / h

        return img, points

    def _resize_for_eval(self, img, points):
        w, h = img.size
        new_w = max(128, (w // 128) * 128)
        new_h = max(128, (h // 128) * 128)

        if new_w == 0 or new_h == 0:
            return img, points

        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            if len(points) > 0:
                points = points.copy()
                points[:, 0] *= new_w / w
                points[:, 1] *= new_h / h

        return img, points

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets