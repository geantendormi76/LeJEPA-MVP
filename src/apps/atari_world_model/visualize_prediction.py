
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
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
# Find latest checkpoint
RUNS_DIR = Path("runs/atari_mvp")
MODEL_PATH = sorted(list(RUNS_DIR.glob("ep*.pth")), key=lambda x: int(x.stem[2:]))[-1]

# Pong Action Map
ACTION_MAP = {
    0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"
}

def visualize():
    print(f"🎨 Visualizing Model: {MODEL_PATH.name}")
    
    # 1. Load Model
    model = AtariWorldModel(img_size=84, num_actions=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. Load Data
    with h5py.File(DATA_PATH, 'r') as f:
        total = len(f['obs'])
        # Pick 5 random samples (Sorted indices for h5py)
        indices = np.sort(np.random.choice(total, 5, replace=False))
        
        obs = torch.from_numpy(f['obs'][indices]).float() / 255.0
        actions = torch.tensor(f['actions'][indices]).long()
        next_obs = torch.from_numpy(f['next_obs'][indices]).float() / 255.0
        
        # Pick 3 random distractors for each sample
        distractors = []
        for _ in range(3):
            rand_idx = np.sort(np.random.choice(total, 5, replace=False))
            d = torch.from_numpy(f['next_obs'][rand_idx]).float() / 255.0
            distractors.append(d)

    # 3. Inference
    obs = obs.to(DEVICE)
    actions = actions.to(DEVICE)
    
    with torch.no_grad():
        out = model(obs, actions)
        pred_vec = out['z_y_pred_feat'] # [5, D]
        
        candidates_vec = []
        candidates_img = []
        
        # Add Truth (Index 0)
        true_vec, _ = model.encode_target(next_obs.to(DEVICE))
        candidates_vec.append(true_vec)
        candidates_img.append(next_obs)
        
        # Add Distractors (Index 1-3)
        for d in distractors:
            d_vec, _ = model.encode_target(d.to(DEVICE))
            candidates_vec.append(d_vec)
            candidates_img.append(d)
            
    # 4. Plotting
    fig, axes = plt.subplots(5, 6, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.1)
    
    for i in range(5): # Rows
        # --- Col 0: Input ---
        img_in = (obs[i, 0].cpu().numpy() * 255).astype(np.uint8)
        axes[i, 0].imshow(img_in, cmap='gray')
        axes[i, 0].set_title("State (t)", fontsize=10)
        axes[i, 0].axis('off')
        
        # --- Col 1: Action ---
        act_id = actions[i].item()
        act_name = ACTION_MAP.get(act_id, str(act_id))
        
        blank = np.zeros((84, 84), dtype=np.uint8)
        blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
        cv2.putText(blank, act_name, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        axes[i, 1].imshow(blank)
        axes[i, 1].set_title(f"Action", fontsize=10)
        axes[i, 1].axis('off')
        
        # --- Col 2-5: Candidates ---
        dists = []
        for c_idx in range(4):
            d = (pred_vec[i] - candidates_vec[c_idx][i]).pow(2).sum().item()
            dists.append(d)
            
        best_idx = np.argmin(dists)
        
        # Shuffle display order to prevent cheating visually (Truth is always 0 in data)
        # But for visualization clarity, we keep Truth at Col 2
        
        for c_idx in range(4):
            ax = axes[i, c_idx + 2]
            img_c = (candidates_img[c_idx][i, 0].cpu().numpy() * 255).astype(np.uint8)
            
            is_truth = (c_idx == 0)
            is_selected = (c_idx == best_idx)
            
            img_c = cv2.cvtColor(img_c, cv2.COLOR_GRAY2RGB)
            
            # Color Logic: Green=Truth, Red=Wrong
            border_color_int = (50, 50, 50)
            if is_truth: border_color_int = (0, 255, 0)
            
            thickness = 1
            if is_selected: 
                thickness = 4
                if is_truth:
                    border_color_int = (0, 255, 0)
                else:
                    border_color_int = (255, 0, 0)
            
            img_c = cv2.copyMakeBorder(img_c, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=border_color_int)
            
            ax.imshow(img_c)
            ax.set_title(f"Dist: {dists[c_idx]:.2f}", fontsize=9, color="green" if is_truth else "black")
            ax.axis('off')
            
            if is_selected:
                # [Fix] Convert int tuple (0-255) to float tuple (0.0-1.0) for Matplotlib
                mpl_color = tuple(c / 255.0 for c in border_color_int)
                ax.text(0.5, -0.1, "PREDICTED", transform=ax.transAxes, ha="center", 
                        color=mpl_color, weight="bold")

    plt.suptitle(f"LeJEPA Prediction Visualization (Ep {MODEL_PATH.stem[2:]})\n(Green Box = True Future | Thick Box = Model Choice)", fontsize=14)
    save_path = "atari_prediction_vis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")

if __name__ == "__main__":
    visualize()
