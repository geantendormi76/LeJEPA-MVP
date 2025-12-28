
import torch
import h5py
import numpy as np
import cv2
import imageio
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
RUNS_DIR = Path("runs/atari_mvp")
MODEL_PATH = sorted(list(RUNS_DIR.glob("ep*.pth")), key=lambda x: int(x.stem[2:]))[-1]
OUTPUT_DIR = Path("runs/atari_mvp/demos")
OUTPUT_DIR.mkdir(exist_ok=True)

def find_ball(img_gray):
    """
    Locate the ball. Returns center (x, y) or None.
    """
    # [Fix] Lower threshold significantly. 
    # Resizing 210->84 makes the white ball (236) blend with black bg, becoming gray (~50-100).
    # We use 30 to catch anything non-black.
    _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_ball = None
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # Filter 1: Score Area (Top 12px)
        if cY < 12: continue
        
        # Filter 2: Paddles are usually at the very edges or bottom
        # But let's be permissive. The ball is usually smaller than paddles.
        # Paddle area is roughly 4-8 pixels in 84x84. Ball is 1-2 pixels.
        
        # We prioritize the object closest to the center of the screen (heuristic)
        # or just return the smallest valid blob.
        if 0 < area < 20: 
            return (cX, cY)
            
    return None

def draw_overlay(img_rgb, action_id, ball_pos=None, title=""):
    vis = img_rgb.copy()
    h, w = vis.shape[:2]
    
    # 1. Draw Ball Highlight
    if ball_pos:
        cv2.circle(vis, ball_pos, 5, (255, 0, 0), 1) 

    # 2. Draw Action Arrow
    center = (w // 2, h // 2)
    color = (0, 255, 0) # Green
    
    if action_id in [2, 4]: # RIGHT
        cv2.arrowedLine(vis, (center[0]-10, center[1]), (center[0]+10, center[1]), color, 2, tipLength=0.5)
        cv2.putText(vis, "RIGHT", (center[0]-15, center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    elif action_id in [3, 5]: # LEFT
        cv2.arrowedLine(vis, (center[0]+10, center[1]), (center[0]-10, center[1]), color, 2, tipLength=0.5)
        cv2.putText(vis, "LEFT", (center[0]-15, center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    elif action_id == 1: # FIRE
        cv2.circle(vis, center, 10, color, 1)
        cv2.putText(vis, "FIRE", (center[0]-10, center[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    else:
        cv2.putText(vis, "NO-OP", (center[0]-15, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # 3. Title
    if title:
        cv2.rectangle(vis, (0, 0), (w, 10), (0, 0, 0), -1)
        cv2.putText(vis, title, (2, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
    return vis

def generate_demo():
    print(f"🎬 Generating Demo GIFs using model: {MODEL_PATH.name}")
    
    model = AtariWorldModel(img_size=84, num_actions=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    f = h5py.File(DATA_PATH, 'r')
    total_len = len(f['obs'])
    
    # [Fix] Force generation even if ball detection is flaky
    print("🔍 Sampling random examples...")
    
    # Just pick 10 random moving examples
    indices = []
    attempts = 0
    while len(indices) < 10 and attempts < 1000:
        idx = np.random.randint(0, total_len)
        act = f['actions'][idx]
        # Prefer moving actions
        if act in [2, 3]: 
            indices.append(idx)
        attempts += 1
        
    if not indices:
        print("⚠️ Could not find moving actions, using random ones.")
        indices = list(np.random.choice(total_len, 10, replace=False))

    print(f"🚀 Generating {len(indices)} GIFs...")

    # Load Distractors
    rand_indices = np.sort(np.random.choice(total_len, 100, replace=False))
    distractors = torch.from_numpy(f['next_obs'][rand_indices]).float() / 255.0
    distractors = distractors.to(DEVICE)

    for i, idx in enumerate(indices):
        # Load Data
        obs = torch.from_numpy(f['obs'][idx:idx+1]).float() / 255.0
        action = torch.tensor(f['actions'][idx:idx+1]).long()
        next_obs = torch.from_numpy(f['next_obs'][idx:idx+1]).float() / 255.0
        
        obs = obs.to(DEVICE)
        action = action.to(DEVICE)
        next_obs = next_obs.to(DEVICE)
        
        # Inference
        with torch.no_grad():
            out = model(obs, action)
            pred_vec = out['z_y_pred_feat']
            truth_vec, _ = model.encode_target(next_obs)
            
            # Retrieval
            min_dist = (pred_vec - truth_vec).pow(2).sum().item()
            best_img = next_obs
            is_correct = True
            
            for k in range(9):
                d_vec, _ = model.encode_target(distractors[k:k+1])
                dist = (pred_vec - d_vec).pow(2).sum().item()
                if dist < min_dist:
                    min_dist = dist
                    best_img = distractors[k:k+1]
                    is_correct = False
        
        # Draw
        img_t0 = (obs[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img_t0_rgb = cv2.cvtColor(img_t0, cv2.COLOR_GRAY2RGB)
        
        # [Debug] Print max pixel value to verify dimming theory
        if i == 0:
            print(f"   [Debug] Max pixel value in frame: {img_t0.max()}")
        
        ball_t0 = find_ball(img_t0)
        
        img_pred = (best_img[0, 0].cpu().numpy() * 255).astype(np.uint8)
        img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_GRAY2RGB)
        ball_pred = find_ball(img_pred)
        
        act_id = action.item()
        scale = 4
        
        f1 = draw_overlay(img_t0_rgb, act_id, ball_t0, title="Input")
        f1 = cv2.resize(f1, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        status = "CORRECT" if is_correct else "WRONG"
        f2 = draw_overlay(img_pred_rgb, -1, ball_pred, title=f"Pred: {status}")
        f2 = cv2.resize(f2, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        # Borders
        col = (0, 255, 0) if is_correct else (0, 0, 255)
        f2 = cv2.copyMakeBorder(f2, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=col)
        f1 = cv2.copyMakeBorder(f1, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(50,50,50))
        
        frames = [f1]*5 + [f2]*5
        save_name = OUTPUT_DIR / f"demo_{i}_{status}.gif"
        imageio.mimsave(save_name, frames, fps=5, loop=0)
        print(f"   💾 Saved: {save_name}")

    f.close()
    print("✨ Done.")

if __name__ == "__main__":
    generate_demo()
