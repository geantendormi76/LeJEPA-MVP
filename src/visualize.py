import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
import re
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.config import cfg
from src.dataset import Galaxy10Dataset
from src.modeling.encoder import LeJEPA_Encoder

def find_best_ckpt(target_ep=450):
    ckpt_dir = cfg.CHECKPOINT_DIR
    # 模糊匹配 ep450
    candidates = list(ckpt_dir.glob(f"*ep{target_ep}.pth"))
    if not candidates:
        print(f"❌ 未找到 Epoch {target_ep} 的模型，请检查路径！")
        # Fallback to latest
        all_ckpts = sorted(list(ckpt_dir.glob("*.pth")), key=lambda x: int(re.search(r'ep(\d+)', x.name).group(1)))
        return all_ckpts[-1]
    return candidates[0]

def run_vis():
    # 1. 加载黄金模型
    ckpt_path = find_best_ckpt(450)
    print(f"🎨 [Vis] 加载黄金模型: {ckpt_path.name}")
    
    device = cfg.DEVICE
    model = LeJEPA_Encoder(
        backbone_name=cfg.BACKBONE,
        img_size=cfg.IMG_SIZE,
        proj_dim=cfg.PROJ_DIM
    ).to(device)
    
    state_dict = torch.load(ckpt_path, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 2. 准备数据 (采样 2000 个点即可，太多图会乱)
    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    full_dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=val_transform)
    
    # 随机采样索引
    indices = np.random.choice(len(full_dataset), 2000, replace=False)
    subset = torch.utils.data.Subset(full_dataset, indices)
    loader = DataLoader(subset, batch_size=128, shuffle=False, num_workers=4)

    # 3. 提取特征
    print("⚡ 正在提取特征...")
    features = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            imgs = imgs.to(device)
            feats, _ = model(imgs) # [B, 384]
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            
    X = np.concatenate(features)
    y = np.concatenate(labels)

    # 4. t-SNE 降维
    print("📉 执行 t-SNE 降维...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # 5. 绘图
    print("🖌️ 正在绘制星系分布图...")
    plt.figure(figsize=(12, 10))
    
    # Galaxy10 类别名
    class_names = [
        "Disk, Face-on, No Spiral", "Smooth, Round", "Smooth, In-between", "Smooth, Cigar",
        "Disk, Edge-on, Rounded", "Disk, Edge-on, Boxy", "Disk, Edge-on, No Bulge",
        "Disk, Face-on, Tight Spiral", "Disk, Face-on, Medium Spiral", "Disk, Face-on, Loose Spiral"
    ]
    
    scatter = sns.scatterplot(
        x=X_embedded[:, 0], y=X_embedded[:, 1],
        hue=[class_names[i] for i in y],
        palette="tab10", s=60, alpha=0.8, edgecolor="w"
    )
    
    plt.title(f"LeJEPA World Model Feature Space (Epoch 450)\nEach point is a Galaxy", fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Galaxy Type")
    plt.tight_layout()
    
    save_path = "vis_ep450_tsne.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 可视化完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    run_vis()