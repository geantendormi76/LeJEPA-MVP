
import gymnasium as gym
import numpy as np
import h5py
import cv2
import os
import sys
from pathlib import Path
from tqdm import tqdm

# [Fix] Explicitly import ale_py for Gym 1.0+ registration
try:
    import ale_py
except ImportError:
    print("❌ Critical: ale_py not found. Please run: pip install ale-py")
    sys.exit(1)

# Config
# [Fix] Use the canonical ID for Gym 1.0+ (though v4 usually works, this is safer)
GAME_ID = "PongNoFrameskip-v4" 
DATA_SIZE = 50000 
SAVE_PATH = Path("data/atari_pong_50k.h5")
IMG_SIZE = 84

def generate():
    if SAVE_PATH.exists():
        print(f"⚠️ Data already exists at {SAVE_PATH}")
        return

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"🎮 Initializing {GAME_ID}...")
    
    # [Fix] Register ALE environments explicitly if needed
    gym.register_envs(ale_py)
    
    try:
        env = gym.make(GAME_ID, render_mode="rgb_array")
    except Exception as e:
        print(f"❌ Gym Error: {e}")
        print("\n💡 Troubleshooting:")
        print("1. Did you run: AutoROM --accept-license ?")
        print("2. Try reinstalling: pip install \"gymnasium[atari]\"")
        return

    env = gym.wrappers.ResizeObservation(env, (IMG_SIZE, IMG_SIZE))
    env = gym.wrappers.GrayscaleObservation(env)
    
    obs_list = []
    action_list = []
    next_obs_list = []
    
    obs, _ = env.reset()
    
    print(f"🚀 Generating {DATA_SIZE} frames (Random Policy)...")
    
    for _ in tqdm(range(DATA_SIZE)):
        action = env.action_space.sample() # Random action
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # Save (S, A, S')
        # obs is (84, 84), we expand to (1, 84, 84) for Channel dim
        obs_list.append(np.expand_dims(obs, 0)) 
        action_list.append(action)
        next_obs_list.append(np.expand_dims(next_obs, 0))
        
        obs = next_obs
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    
    print("💾 Saving to H5 (Compressed)...")
    with h5py.File(SAVE_PATH, "w") as f:
        f.create_dataset("obs", data=np.array(obs_list, dtype=np.uint8), compression="gzip")
        f.create_dataset("actions", data=np.array(action_list, dtype=np.uint8))
        f.create_dataset("next_obs", data=np.array(next_obs_list, dtype=np.uint8), compression="gzip")
        
    print(f"✅ Dataset ready: {SAVE_PATH}")

if __name__ == "__main__":
    generate()
