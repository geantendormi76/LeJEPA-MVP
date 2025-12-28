
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Path Hack
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.apps.atari_world_model.modeling import AtariWorldModel

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/atari_pong_50k.h5"
MODEL_PATH = "runs/atari_mvp/ep20.pth" # Load the last checkpoint

def verify():
    print("========================================")
    print("   🌌 LeJEPA Physics Verification")
    print("========================================")
    
    # 1. Load Model
    print(f"📂 Loading Model: {MODEL_PATH}")
    if not Path(MODEL_PATH).exists():
        print(f"❌ Model not found at {MODEL_PATH}. Did you finish training?")
        return

    model = AtariWorldModel(img_size=84, num_actions=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. Load Data (Sample 100 pairs)
    print(f"📂 Loading Data Sample...")
    with h5py.File(DATA_PATH, 'r') as f:
        total_len = len(f['obs'])
        
        # [Fix] h5py requires indices to be sorted!
        # Randomly select 100 indices and SORT them
        indices = np.sort(np.random.choice(total_len, 100, replace=False))
        
        obs = torch.from_numpy(f['obs'][indices]).float() / 255.0
        actions = torch.tensor(f['actions'][indices]).long()
        next_obs = torch.from_numpy(f['next_obs'][indices]).float() / 255.0
        
        # Load some random frames for negative samples (Also sorted)
        rand_indices = np.sort(np.random.choice(total_len, 100, replace=False))
        random_obs = torch.from_numpy(f['next_obs'][rand_indices]).float() / 255.0

    obs = obs.to(DEVICE)
    actions = actions.to(DEVICE)
    next_obs = next_obs.to(DEVICE)
    random_obs = random_obs.to(DEVICE)

    # 3. Inference
    print("⚡ Running Inference...")
    with torch.no_grad():
        # Predict Future
        out = model(obs, actions)
        pred_feat = out['z_y_pred_feat'] # [100, D]
        
        # Encode True Future
        true_feat, _ = model.encode_target(next_obs) # [100, D]
        
        # Encode Random Future (Baseline)
        rand_feat, _ = model.encode_target(random_obs) # [100, D]

    # 4. Compute Distances (MSE)
    # Distance to Truth
    dist_true = (pred_feat - true_feat).pow(2).sum(dim=1) # [100]
    
    # Distance to Random
    dist_rand = (pred_feat - rand_feat).pow(2).sum(dim=1) # [100]
    
    # 5. Analysis
    correct_count = (dist_true < dist_rand).sum().item()
    accuracy = correct_count / 100.0
    
    avg_dist_true = dist_true.mean().item()
    avg_dist_rand = dist_rand.mean().item()
    
    print("\n📊 Verification Results:")
    print(f"   - Avg Distance to TRUE Future:   {avg_dist_true:.4f} (Lower is better)")
    print(f"   - Avg Distance to RANDOM Frame:  {avg_dist_rand:.4f}")
    print(f"   - Contrastive Margin:            {avg_dist_rand - avg_dist_true:.4f}")
    print("-" * 40)
    print(f"🏆 Physics Understanding Score: {accuracy * 100:.1f}%")
    print("(Interpretation: The model correctly identified the real future vs a random frame X% of the time)")
    
    if accuracy > 0.8:
        print("\n✅ SUCCESS: The model understands causality!")
    elif accuracy > 0.6:
        print("\n⚠️ WARNING: Weak understanding. Needs more training or data.")
    else:
        print("\n❌ FAILURE: The model is guessing.")

if __name__ == "__main__":
    verify()
