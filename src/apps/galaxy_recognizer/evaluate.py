import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import sys
import re
from pathlib import Path

# 路径 Hack
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.config import cfg
from src.dataset import Galaxy10Dataset
from src.modeling.encoder import LeJEPA_Encoder

def load_model(ckpt_path, device):
    """加载指定 Checkpoint 的模型"""
    model = LeJEPA_Encoder(
        backbone_name=cfg.BACKBONE,
        img_size=cfg.IMG_SIZE,
        proj_dim=cfg.PROJ_DIM
    ).to(device)
    
    # 处理 torch.compile 带来的前缀问题
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.eval()
    return model

def extract_features(model, loader, device):
    """提取全量特征"""
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            # LeJEPA forward 返回 (embedding, projection)
            # 我们只需要 embedding (384维)
            feats, _ = model(imgs)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

def run_sweep():
    print("🔍 [AEGIS] 启动线性探测扫描 (Linear Probing Sweep)...")
    
    # 1. 准备数据 (只做标准化，不做强增强)
    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=val_transform)
    # 使用大 Batch 加速推理
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=8)
    
    # 2. 扫描所有 Checkpoints
    ckpt_dir = cfg.CHECKPOINT_DIR
    # 按 Epoch 数字排序 (ep50, ep100...)
    ckpts = sorted(list(ckpt_dir.glob("*.pth")), key=lambda x: int(re.search(r'ep(\d+)', x.name).group(1)))
    
    if not ckpts:
        print("❌ 未找到权重文件！")
        return

    results = []
    
    print(f"📂 发现 {len(ckpts)} 个模型存档，开始逐一评估...")
    print("-" * 60)
    print(f"{'Epoch':<10} | {'Accuracy':<10} | {'Status'}")
    print("-" * 60)

    best_acc = 0.0
    best_ep = 0

    for ckpt in ckpts:
        epoch_num = int(re.search(r'ep(\d+)', ckpt.name).group(1))
        
        # 加载模型
        model = load_model(ckpt, cfg.DEVICE)
        
        # 提取特征
        X, y = extract_features(model, loader, cfg.DEVICE)
        
        # 划分训练/测试集 (80/20)
        # 注意：这里是在冻结的特征上训练分类器
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 训练逻辑回归 (秒级)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        clf.fit(X_train, y_train)
        
        # 评估
        acc = accuracy_score(y_test, clf.predict(X_test)) * 100
        results.append((epoch_num, acc))
        
        if acc > best_acc:
            best_acc = acc
            best_ep = epoch_num
            status = "🔥 New Best!"
        else:
            status = ""
            
        print(f"{epoch_num:<10} | {acc:.2f}%      | {status}")

    # 3. 总结与绘图
    print("-" * 60)
    print(f"🏆 最佳模型: Epoch {best_ep} (Acc: {best_acc:.2f}%)")
    
    epochs, accs = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accs, marker='o', linestyle='-', color='b', linewidth=2)
    plt.title(f"LeJEPA Training Progress (Linear Probe Accuracy)\nBest: {best_acc:.2f}% @ Ep{best_ep}", fontsize=14)
    plt.xlabel("Training Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    
    # 标记最高点
    plt.plot(best_ep, best_acc, 'r*', markersize=15)
    
    save_path = "evaluation_curve.png"
    plt.savefig(save_path, dpi=300)
    print(f"📊 评估曲线已保存至: {save_path}")

if __name__ == "__main__":
    run_sweep()