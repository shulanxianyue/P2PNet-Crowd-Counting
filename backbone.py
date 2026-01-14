import torch
import torch.nn as nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        features = list(backbone.features.children())
        
        # VGG16_BN's level indexs：
        # Pool 1: idx 6
        # Pool 2: idx 13
        # Pool 3: idx 23 
        # Pool 4: idx 33
        # Pool 5: idx 43
        
        self.body1 = nn.Sequential(*features[:24])   # 输出 Stride 8 (包含了 idx 23 的 Pool)
        self.body2 = nn.Sequential(*features[24:34]) # 输出 Stride 16 (包含了 idx 33 的 Pool)
        self.body3 = nn.Sequential(*features[34:44]) # 输出 Stride 32 (包含了 idx 43 的 Pool)
        
        # freeze Batch Norm level (prevent overfitting)
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, x):
        out = []
        
        # 1. body1 -> get Stride 8 feature (C3)
        x = self.body1(x)
        out.append(x)
        
        # 2. body2 -> get Stride 16 feature (C4)
        x = self.body2(x)
        out.append(x)
        
        # 3. body3 -> get Stride 32 feature (C5)
        x = self.body3(x)
        out.append(x)
        
        return out