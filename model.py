import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # Load VGG16 with Batch Normalization
        # We only need the features, not the classifier
        vgg = models.vgg16_bn(pretrained=True)
        self.features = vgg.features
        
        # In the official P2PNet implementation, they use features from
        # specific layers to form a Feature Pyramid.
        # Layer indices for VGG16_BN:
        # 23 -> Relu after Conv3_3 (Stride 4) - Optional usually
        # 33 -> Relu after Conv4_3 (Stride 8)
        # 43 -> Relu after Conv5_3 (Stride 16)
        
        # We will focus on Stride 8 and Stride 16 features
        self.layer_stride8 = nn.Sequential(*list(self.features.children())[:33])
        self.layer_stride16 = nn.Sequential(*list(self.features.children())[33:43])

    def forward(self, x):
        feat8 = self.layer_stride8(x)
        feat16 = self.layer_stride16(feat8)
        return feat8, feat16

class P2PNet(nn.Module):
    def __init__(self, backbone_name='vgg16_bn'):
        super(P2PNet, self).__init__()
        self.backbone = Backbone()
        
        # Lateral layers to reduce channel dimensions
        # VGG Conv4_3 has 512 channels, Conv5_3 has 512 channels
        self.conv_stride8 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv_stride16 = nn.Conv2d(512, 128, kernel_size=1)

        # Regression Head: Predicts (x, y) offsets
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 2, kernel_size=1) # Output: dx, dy
        )

        # Classification Head: Predicts confidence score (Foreground vs Background)
        self.classification_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1) # Output: score
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def create_grid(self, h, w, device):
        # Create a grid of reference points (anchor points)
        # P2PNet predicts offsets relative to these grid points.
        stride = 8 
        shift_x = torch.arange(0, w, dtype=torch.float32, device=device) * stride
        shift_y = torch.arange(0, h, dtype=torch.float32, device=device) * stride
        
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        shift_x = shift_x.flatten()
        shift_y = shift_y.flatten()
        
        # Add half stride to center the points
        shift_x += stride // 2
        shift_y += stride // 2
        
        return torch.stack((shift_x, shift_y), dim=1) # Shape: (H*W, 2)

    def forward(self, x):
        # 1. Feature Extraction
        # feat8: [B, 512, H/8, W/8]
        # feat16: [B, 512, H/16, W/16]
        feat8, feat16 = self.backbone(x)
        
        # 2. Feature Fusion (FPN style)
        feat8_proj = self.conv_stride8(feat8)
        feat16_proj = self.conv_stride16(feat16)
        
        # Upsample stride 16 to stride 8 and concatenate
        feat16_up = F.interpolate(feat16_proj, size=feat8.shape[2:], mode='nearest')
        
        # Merged features: [B, 256, H/8, W/8]
        feat_merged = torch.cat([feat8_proj, feat16_up], dim=1)
        
        # 3. Prediction Heads
        # pred_logits: [B, 1, H/8, W/8]
        # pred_offsets: [B, 2, H/8, W/8]
        pred_logits = self.classification_head(feat_merged)
        pred_offsets = self.regression_head(feat_merged)
        
        # 4. Format Output
        B, _, H, W = pred_logits.shape
        
        # Flatten spatial dimensions: [B, C, H, W] -> [B, H*W, C]
        pred_logits = pred_logits.flatten(2).permute(0, 2, 1) # [B, N, 1]
        pred_offsets = pred_offsets.flatten(2).permute(0, 2, 1) # [B, N, 2]
        
        # 5. Decode Coordinates
        # The network predicts offsets relative to the grid center.
        # Absolute Pos = Grid Pos + Predicted Offset
        grid_points = self.create_grid(H, W, x.device) # [N, 2]
        pred_points = grid_points.unsqueeze(0) + pred_offsets # [B, N, 2]
        
        return {
            'pred_logits': pred_logits, # Classification scores
            'pred_points': pred_points  # Absolute coordinates (x, y)
        }

if __name__ == "__main__":
    # Sanity Check
    print("Initializing P2PNet...")
    model = P2PNet()
    model.eval()
    
    # Create a dummy input tensor: Batch=2, Channels=3, Height=224, Width=224
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print(f"Input Shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    logits = output['pred_logits']
    points = output['pred_points']
    
    print("\n--- Output Check ---")
    print(f"Logits Shape: {logits.shape} (Expected: [2, 784, 1])")
    print(f"Points Shape: {points.shape} (Expected: [2, 784, 2])")
    
    # 784 comes from (224/8) * (224/8) = 28 * 28
    if logits.shape[1] == 28 * 28:
        print("\n✅ Model structure verification successful.")
    else:
        print("\n❌ Dimension mismatch.")