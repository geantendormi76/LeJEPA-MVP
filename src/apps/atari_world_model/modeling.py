
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Path Hack
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.encoder import LeJEPA_Encoder

class LeJEPA_Predictor(nn.Module):
    def __init__(self, embed_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.action_embed = nn.Embedding(action_dim, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, z_state, action_ids):
        z_action = self.action_embed(action_ids) 
        z_input = torch.cat([z_state, z_action], dim=1)
        z_pred = self.net(z_input) 
        return z_pred

class AtariWorldModel(nn.Module):
    def __init__(self, img_size=84, num_actions=6, proj_dim=256):
        super().__init__()
        
        # [Strategy: ResNet] Better for small pixel objects in Atari
        self.encoder = LeJEPA_Encoder(
            backbone_name="resnet18", 
            img_size=img_size,
            proj_dim=proj_dim
        )
        
        # [Hack] Modify first conv for Grayscale (1 channel)
        backbone = self.encoder.backbone
        if hasattr(backbone, 'conv1'):
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                               stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias)
            with torch.no_grad():
                new_conv.weight[:] = old_conv.weight.sum(dim=1, keepdim=True) / 3.0
            backbone.conv1 = new_conv
        
        embed_dim = self.encoder.backbone.num_features
        self.predictor = LeJEPA_Predictor(embed_dim=embed_dim, action_dim=num_actions)
        
    def forward(self, x, action):
        z_x_feat, z_x_proj = self.encoder(x)
        z_y_pred_feat = self.predictor(z_x_feat, action)
        z_y_pred_proj = self.encoder.projector(z_y_pred_feat)
        
        return {
            "z_x_proj": z_x_proj,
            "z_x_feat": z_x_feat,
            "z_y_pred_feat": z_y_pred_feat,
            "z_y_pred_proj": z_y_pred_proj
        }
        
    def encode_target(self, y):
        with torch.no_grad():
            z_y_feat, z_y_proj = self.encoder(y)
        return z_y_feat, z_y_proj
