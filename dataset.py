import torch
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

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
            
        # 拼凑路径
        # 假设 root 是 .../expd
        path_img = os.path.join(root, part, sub_dir, 'images')
        path_gt = os.path.join(root, part, sub_dir, 'ground_truth')

        # 2. 获取所有图片文件名
        # 只读取文件名，不包含路径
        if not os.path.exists(path_img):
            raise FileNotFoundError(f"Image path not found: {path_img}")
        if not os.path.exists(path_gt):
            raise FileNotFoundError(f"GT path not found: {path_gt}")

        img_names = [f for f in os.listdir(path_img) if f.endswith('.jpg')]
        
        self.img_paths = []
        self.gt_paths = []

        # 3. 构造完整路径 & 匹配 .npy 文件
        for img_name in img_names:
            # 图片完整路径
            self.img_paths.append(os.path.join(path_img, img_name))
            
            # 对应的 GT 文件名: IMG_10.jpg -> IMG_10.npy (根据你的描述)
            gt_name = img_name.replace('.jpg', '.npy')
            
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
        
        # --- 修改点：加载 .npy 文件 ---
        try:
            points = np.load(gt_path) # Shape: [N, 2]
        except Exception as e:
            print(f"Error loading {gt_path}: {e}")
            points = np.zeros((0, 2))

        points = torch.tensor(points, dtype=torch.float32) # (x, y)

        # 2. 预处理 (Train vs Test)
        if self.method == 'train':
            # --- 训练：随机裁剪固定大小 (224x224) ---
            i, j, th, tw = self._random_crop_params(h, w, self.crop_size, self.crop_size)
            
            # Crop 图片
            img = img.crop((j, i, j + tw, i + th))
            
            # Crop 点 (坐标平移)
            if len(points) > 0:
                points[:, 0] -= j
                points[:, 1] -= i
                # 过滤掉跑出裁剪框的点
                mask = (points[:, 0] >= 0) & (points[:, 0] < tw) & \
                       (points[:, 1] >= 0) & (points[:, 1] < th)
                points = points[mask]
        
        else:
            # --- 测试/验证：调整为 128 倍数 (Lanczos Resizing) ---
            new_w = (w // 128) * 128
            new_h = (h // 128) * 128
            
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # 调整点坐标
            scale_x = new_w / w
            scale_y = new_h / h
            if len(points) > 0:
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y

        # 3. 归一化 (Standard ImageNet mean/std)
        img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        return img_tensor, {'point': points}

    def _random_crop_params(self, h, w, th, tw):
        if w <= tw or h <= th:
            # 如果图片比裁剪尺寸小，就不裁剪了，直接返回原尺寸（或者这里应该做 padding，简单起见直接返回）
            return 0, 0, h, w
        
        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        return i, j, th, tw

def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets