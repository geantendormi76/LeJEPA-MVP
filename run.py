
import sys
import os

# 确保 src 在路径中
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from train import train

if __name__ == "__main__":
    print("========================================")
    print("   🌌 LeJEPA-Galaxy MVP Launcher")
    print("========================================")
    train()
