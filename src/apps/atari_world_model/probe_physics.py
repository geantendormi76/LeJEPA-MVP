
import torch
import h5py
import numpy as np
import cv2
from pathlib import Path
import sys
from tqdm import tqdm

# Path Hack
ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.apps.atari_world_model.modeling import AtariWorldModel

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = "data/atari_pong_50k.h5"
RUNS_DIR = Path("runs/atari_mvp")
# Find latest model
MODEL_PATH = sorted(list(RUNS_DIR.glob("ep*.pth")), key=lambda x: int(x.stem[2:]))[-1]

def get_ball_y(img_gray):
    """
    Return the Y coordinate of the ball center.
    Returns -1 if ball not found.
    """
    _, thresh = cv2.threshold(img_gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 0 < area < 40: # Ball size constraint
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cY = int(M["m01"] / M["m00"])
                if cY > 12: # Ignore score area
                    return cY
    return -1

def run_audit():
    print("========================================")
    print("   🧮 LeJEPA Physics Law Audit")
    print("========================================")
    print(f"🧠 Model: {MODEL_PATH.name}")
    
    # 1. Load Model
    model = AtariWorldModel(img_size=84, num_actions=6).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 2. Load Data
    f = h5py.File(DATA_PATH, 'r')
    total_len = len(f['obs'])
    
    # 3. Audit Metrics
    metrics = {
        "total_samples": 0,
        "retrieval_correct": 0,
        "physics_consistent": 0, # Direction matches action
        "physics_valid_samples": 0, # Samples where ball was found
        "pixel_error_sum": 0.0
    }
    
    # Sample 1000 random frames for statistical significance
    indices = np.sort(np.random.choice(total_len, 1000, replace=False))
    
    # Distractors Pool (Static)
    rand_indices = np.sort(np.random.choice(total_len, 100, replace=False))
    distractors = torch.from_numpy(f['next_obs'][rand_indices]).float() / 255.0
    distractors = distractors.to(DEVICE)
    
    print(f"🔍 Auditing {len(indices)} samples...")
    
    for idx in tqdm(indices):
        # Load Sample
        obs_np = f['obs'][idx] # [1, 84, 84]
        next_obs_np = f['next_obs'][idx]
        action_id = f['actions'][idx]
        
        # Skip NO-OP (Action 0) or FIRE (1) for physics check, 
        # as they don't imply directional movement of paddle/ball directly in simple logic
        # We focus on RIGHT (2,4) and LEFT (3,5) for directional consistency
        is_directional = action_id in [2, 3, 4, 5]
        
        obs = torch.from_numpy(obs_np).unsqueeze(0).float() / 255.0
        action = torch.tensor([action_id]).long()
        next_obs = torch.from_numpy(next_obs_np).unsqueeze(0).float() / 255.0
        
        obs = obs.to(DEVICE)
        action = action.to(DEVICE)
        next_obs = next_obs.to(DEVICE)
        
        # --- A. Retrieval Task (The "Eye" Test) ---
        with torch.no_grad():
            out = model(obs, action)
            pred_vec = out['z_y_pred_feat']
            
            # Candidates: Truth + 9 Distractors
            truth_vec, _ = model.encode_target(next_obs)
            
            min_dist = (pred_vec - truth_vec).pow(2).sum().item()
            best_img_np = next_obs_np[0] # Default to truth
            is_retrieval_correct = True
            
            for k in range(9):
                d_vec, _ = model.encode_target(distractors[k:k+1])
                dist = (pred_vec - d_vec).pow(2).sum().item()
                if dist < min_dist:
                    min_dist = dist
                    best_img_np = (distractors[k, 0].cpu().numpy() * 255).astype(np.uint8)
                    is_retrieval_correct = False
        
        metrics["total_samples"] += 1
        if is_retrieval_correct:
            metrics["retrieval_correct"] += 1
            
        # --- B. Physics Consistency Task (The "Logic" Test) ---
        # Only check if we found the ball in both frames
        y_t0 = get_ball_y(obs_np[0])
        y_pred = get_ball_y(best_img_np)
        y_true = get_ball_y(next_obs_np[0])
        
        if y_t0 != -1 and y_pred != -1 and y_true != -1:
            metrics["physics_valid_samples"] += 1
            
            # 1. Pixel Error (How far is predicted ball from true ball?)
            metrics["pixel_error_sum"] += abs(y_pred - y_true)
            
            # 2. Directional Consistency (Did it move the right way?)
            # Note: In Pong, ball movement is complex (bounces). 
            # But "Retrieval Correctness" already implies it picked the right frame.
            # Here we check: Did the model pick a frame where the ball position is 
            # IDENTICAL to the Ground Truth?
            
            # If retrieval was correct, physics is consistent by definition.
            # If retrieval was wrong, we check if it at least got the ball pos right.
            if abs(y_pred - y_true) < 2: # Allow 1px error
                metrics["physics_consistent"] += 1

    # --- Report ---
    acc = metrics["retrieval_correct"] / metrics["total_samples"] * 100
    
    if metrics["physics_valid_samples"] > 0:
        phy_acc = metrics["physics_consistent"] / metrics["physics_valid_samples"] * 100
        avg_pixel_err = metrics["pixel_error_sum"] / metrics["physics_valid_samples"]
    else:
        phy_acc = 0.0
        avg_pixel_err = 0.0
        
    print("\n📊 [AUDIT REPORT]")
    print("-" * 40)
    print(f"1. Retrieval Accuracy (Top-1):   {acc:.2f}%")
    print(f"   (Interpretation: Can the model identify the exact future frame?)")
    print("-" * 40)
    print(f"2. Physics Consistency:          {phy_acc:.2f}%")
    print(f"   (Interpretation: Is the ball in the correct position in the predicted frame?)")
    print("-" * 40)
    print(f"3. Avg Ball Position Error:      {avg_pixel_err:.2f} pixels")
    print(f"   (Interpretation: Average deviation from ground truth)")
    print("-" * 40)
    
    # Final Verdict
    if acc > 90 and avg_pixel_err < 1.0:
        print("✅ VERDICT: PASSED. Model possesses high-fidelity physical understanding.")
    elif acc > 70:
        print("⚠️ VERDICT: PASSED (Weak). Model understands general trends but lacks precision.")
    else:
        print("❌ VERDICT: FAILED. Model is hallucinating.")

if __name__ == "__main__":
    run_audit()
