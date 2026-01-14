import torch
import torch.nn as nn
from typing import Optional, List
from torch import Tensor

# --- 1. NestedTensor ---
class NestedTensor(object):
    """
    Motif: The "Custom Shipping Box"
    Images have different sizes. You can't stack them directly into a [B, C, H, W] tensor
    unless you pad them. This class holds the padded tensor AND a mask.
    The mask tells the network: "Pixels here are real image, pixels there are just padding."
    """
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors # The actual image data (padded)
        self.mask = mask       # Binary mask (0=pixel, 1=padding)

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# --- 2. Collate Function ---
def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """
    Helper function to pad images to the largest size in the batch
    and create the mask.
    """
    if tensor_list[0].ndim == 3:
        # Calculate max height and width in this batch
        max_size = [0, 0, 0]
        for img in tensor_list:
            for i in range(3):
                max_size[i] = max(max_size[i], img.shape[i])
        
        # Batch shape: [Batch_Size, C, Max_H, Max_W]
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        
        # Initialize with zeros (black padding)
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        
        # Place images into the top-left corner
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            # Set mask=False for real pixels (confusing naming in DETR, but standard here)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def collate_fn(batch):
    """
    The "Logistics Manager".
    Pytorch's default collate fails if images have different sizes.
    We use this custom one.
    """
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

# --- 3. AverageMeter ---
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count