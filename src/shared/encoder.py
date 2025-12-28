
import torch
import torch.nn as nn
import timm
from torchvision.ops import MLP

class LeJEPA_Encoder(nn.Module):
    def __init__(self, backbone_name="vit_small_patch16_224", img_size=224, proj_dim=256, pretrained=False):
        super().__init__()
        
        # [Fix] Construct arguments dynamically based on architecture type
        model_kwargs = {
            "pretrained": pretrained,
            "num_classes": 0, 
            "drop_path_rate": 0.1 # Supported by both ViT and ResNet (Stochastic Depth)
        }
        
        # ViT needs img_size for Position Embeddings; ResNet (CNN) does not.
        if "vit" in backbone_name:
            model_kwargs["img_size"] = img_size
            
        # 1. Create Backbone
        self.backbone = timm.create_model(backbone_name, **model_kwargs)
        
        # Get Embedding Dimension
        embed_dim = self.backbone.num_features

        # 2. Projector (Aligned with LeJEPA paper)
        self.projector = MLP(
            in_channels=embed_dim,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer=nn.LayerNorm,
            activation_layer=nn.GELU
        )

    def forward(self, x):
        embedding = self.backbone(x)
        projection = self.projector(embedding)
        return embedding, projection
