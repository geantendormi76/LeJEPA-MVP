
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Path Hack
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.shared.loss import SIGRegLoss
from src.apps.atari_world_model.modeling import AtariWorldModel

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 100  # [Full Training] Increased to 100
LR = 1e-3
DATA_PATH = "data/atari_pong_50k.h5"
OUTPUT_DIR = Path("runs/atari_mvp")

class AtariDataset(Dataset):
    def __init__(self, path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Data not found: {path}")
        print(f"📂 Loading H5: {path} ...")
        with h5py.File(path, 'r') as f:
            self.obs = f['obs'][:]
            self.actions = f['actions'][:]
            self.next_obs = f['next_obs'][:]
            
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.obs[idx]).float() / 255.0
        y = torch.from_numpy(self.next_obs[idx]).float() / 255.0
        a = torch.tensor(self.actions[idx]).long()
        return x, a, y

def train():
    print("========================================")
    print("   🌌 LeJEPA Atari World Model (Full Run)")
    print("========================================")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    dataset = AtariDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    model = AtariWorldModel(img_size=84, num_actions=6).to(DEVICE)
    sigreg = SIGRegLoss().to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    mse_loss = torch.nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        total_pred = 0
        total_reg = 0
        
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for x, a, y in pbar:
            x, a, y = x.to(DEVICE), a.to(DEVICE), y.to(DEVICE)
            
            out = model(x, a)
            z_y_true_feat, _ = model.encode_target(y)
            
            # Loss
            loss_pred = mse_loss(out['z_y_pred_feat'], z_y_true_feat)
            loss_reg = sigreg(out['z_x_proj'])
            
            # [Strategy] High SIGReg weight to force structure
            loss = loss_pred + 1.0 * loss_reg
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pred += loss_pred.item()
            total_reg += loss_reg.item()
            
            pbar.set_postfix({
                "Pred": f"{loss_pred.item():.4f}", 
                "Reg": f"{loss_reg.item():.4f}"
            })
            
        print(f"📉 Ep {epoch+1} | Pred: {total_pred/len(loader):.4f} | Reg: {total_reg/len(loader):.4f}")
        
        # Save every 10 epochs
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), OUTPUT_DIR / f"ep{epoch+1}.pth")

if __name__ == "__main__":
    train()
