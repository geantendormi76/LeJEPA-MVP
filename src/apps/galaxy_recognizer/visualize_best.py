
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
from pathlib import Path

# Path Hack
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from configs.config import cfg
from src.apps.galaxy_recognizer.dataset import Galaxy10Dataset
from src.shared.encoder import LeJEPA_Encoder

# [Config] Path to your best model (Based on your screenshot)
BEST_MODEL_PATH = Path("runs/release/best_finetuned_model.pth")

def load_best_encoder(device):
    print(f"📂 Loading Best Model: {BEST_MODEL_PATH}")
    
    if not BEST_MODEL_PATH.exists():
        # Fallback search
        print(f"⚠️ Path not found at {BEST_MODEL_PATH}, searching current dir...")
        candidates = list(Path(".").rglob("best_finetuned_model.pth"))
        if not candidates:
            raise FileNotFoundError("Could not find best_finetuned_model.pth")
        model_path = candidates[0]
    else:
        model_path = BEST_MODEL_PATH

    # Initialize pure Encoder
    encoder = LeJEPA_Encoder(
        backbone_name=cfg.BACKBONE,
        img_size=cfg.IMG_SIZE,
        proj_dim=cfg.PROJ_DIM
    ).to(device)
    
    # Load weights (Surgical extraction)
    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = {}
    
    print("🔧 Adapting weights (Removing classification head)...")
    for k, v in state_dict.items():
        # Finetuned model keys look like: "encoder.backbone.xxx" or "head.weight"
        # We need: "backbone.xxx"
        if k.startswith("encoder."):
            new_key = k.replace("encoder.", "")
            new_state_dict[new_key] = v
            
    # Load (strict=False to ignore missing 'head' weights, which is intentional)
    msg = encoder.load_state_dict(new_state_dict, strict=False)
    print(f"✅ Weights loaded. Missing keys (expected 'head'): {len(msg.missing_keys)}")
    
    encoder.eval()
    return encoder

def run_contrastive_search():
    print("🔍 [SOTA Vis] Generating Contrastive Search with BEST Model...")
    device = cfg.DEVICE
    
    # 1. Load Model
    encoder = load_best_encoder(device)

    # 2. Prepare Data
    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=None)
    proc_dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=val_transform)
    loader = DataLoader(proc_dataset, batch_size=128, shuffle=False, num_workers=4)

    # 3. Build Feature Index
    print("⚡ Building Feature Index (This may take a moment)...")
    features_db = []
    labels_db = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            feats, _ = encoder(imgs)
            # Normalize for Cosine Similarity
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            features_db.append(feats.cpu())
            labels_db.append(labels)
            
    features_db = torch.cat(features_db, dim=0)
    labels_db = torch.cat(labels_db, dim=0).numpy()

    # 4. Select Queries (Representative Classes)
    target_classes = {
        "Spiral (旋涡星系)": 7,  
        "Smooth (圆滑星系)": 1, 
        "Edge-on (侧向星系)": 4   
    }
    
    # 5. Plotting
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for row, (name, label_idx) in enumerate(target_classes.items()):
        # Find a random query of this class
        indices = np.where(labels_db == label_idx)[0]
        q_idx = np.random.choice(indices)
        
        # Search
        query_feat = features_db[q_idx].unsqueeze(0)
        sim_scores = torch.mm(query_feat, features_db.t()).squeeze(0)
        
        # Get Top-6 (Top-1 is self)
        topk_scores, topk_indices = torch.topk(sim_scores, k=6)
        
        # Draw Query
        query_img, _ = raw_dataset[q_idx]
        axes[row, 0].imshow(query_img)
        axes[row, 0].set_title(f"QUERY\n{name}", color="darkred", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Draw Matches
        for col in range(1, 6):
            idx = topk_indices[col].item()
            score = topk_scores[col].item()
            match_img, _ = raw_dataset[idx]
            
            axes[row, col].imshow(match_img)
            axes[row, col].set_title(f"Sim: {score:.3f}", fontsize=10)
            axes[row, col].axis('off')

    plt.suptitle(f"LeJEPA Full-Training Result (Best Model)\nNotice the semantic consistency!", fontsize=16, y=0.98)
    
    save_path = "best_model_contrastive.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ Image saved to: {save_path}")

if __name__ == "__main__":
    run_contrastive_search()
