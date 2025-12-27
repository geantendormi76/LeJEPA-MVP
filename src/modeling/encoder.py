
import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP

class LeJEPA_Encoder(nn.Module):
    def __init__(self, backbone_name="vit_small_patch16_224", img_size=224, proj_dim=256, pretrained=False):
        super().__init__()
        
        # 1. 使用 ViT 作为主干
        # LeJEPA 在 ViT 上能学到更好的层次化表征
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0, 
            img_size=img_size,
            drop_path_rate=0.1 # 增加随机深度，防止过拟合
        )
        
        # 获取 Embedding 维度 (ViT-Small 是 384)
        embed_dim = self.backbone.num_features

        # 2. Projector (对齐论文 3 层架构)
        self.projector = MLP(
            in_channels=embed_dim,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer=nn.LayerNorm, # ViT 体系惯用 LayerNorm
            activation_layer=nn.GELU     # ViT 体系惯用 GELU
        )

    def forward(self, x):
        embedding = self.backbone(x)
        projection = self.projector(embedding)
        return embedding, projection
