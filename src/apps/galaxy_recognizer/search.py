
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys
from pathlib import Path

# 路径 Hack
sys.path.append(str(Path(__file__).resolve().parents[1]))

from configs.config import cfg
from src.dataset import Galaxy10Dataset
from src.modeling.encoder import LeJEPA_Encoder

def run_contrastive_search():
    print("🔍 [Demo] 启动差异化对比搜索 (Contrastive Search)...")
    
    # 1. 加载模型
    ckpt_dir = cfg.CHECKPOINT_DIR
    ckpts = sorted(list(ckpt_dir.glob("*.pth")))
    if not ckpts:
        print("❌ 未找到权重文件")
        return
    latest_ckpt = ckpts[-1]
    print(f"📂 加载权重: {latest_ckpt.name}")

    device = cfg.DEVICE
    encoder = LeJEPA_Encoder(
        backbone_name=cfg.BACKBONE,
        img_size=cfg.IMG_SIZE,
        proj_dim=cfg.PROJ_DIM
    ).to(device)
    
    state_dict = torch.load(latest_ckpt, map_location=device)
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    encoder.load_state_dict(new_state_dict)
    encoder.eval()

    # 2. 准备数据
    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 原始数据集用于显示
    raw_dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=None)
    # 处理后的数据集用于推理
    proc_dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=val_transform)
    
    # 使用较大的 Batch 加速特征提取
    loader = DataLoader(proc_dataset, batch_size=256, shuffle=False, num_workers=8)

    # 3. 提取特征库 & 标签索引
    print("⚡ 正在构建全量特征库...")
    features_db = []
    labels_db = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs = imgs.to(device)
            feats, _ = encoder(imgs)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            features_db.append(feats.cpu())
            labels_db.append(labels)
            
    features_db = torch.cat(features_db, dim=0)
    labels_db = torch.cat(labels_db, dim=0).numpy()
    print(f"📚 索引库构建完成: {features_db.shape}")

    # 4. 定向筛选 Query (Hardcoded Selection)
    # Galaxy10 类别映射:
    # 0: Disk, Face-on, No Spiral
    # 1: Smooth, Completely round (圆蛋)
    # 2: Smooth, in-between
    # 3: Smooth, Cigar shaped
    # 4: Disk, Edge-on, Rounded Bulge (侧向/飞碟)
    # 5: Disk, Edge-on, Boxy Bulge
    # 6: Disk, Edge-on, No Bulge
    # 7: Disk, Face-on, Tight Spiral (螺旋)
    # 8: Disk, Face-on, Medium Spiral
    # 9: Disk, Face-on, Loose Spiral

    # 我们选择最具代表性的三类
    target_classes = {
        "Spiral (旋涡状)": 7,  # 螺旋
        "Smooth (圆蛋状)": 1,  # 光滑圆
        "Edge-on (飞碟状)": 4   # 侧向
    }
    
    query_indices = []
    query_titles = []
    
    # 从数据库中为每一类找一个代表
    for name, label_idx in target_classes.items():
        # 找到所有属于该类别的索引
        indices = np.where(labels_db == label_idx)[0]
        if len(indices) > 0:
            # 随机选一个作为 Query
            selected_idx = np.random.choice(indices)
            query_indices.append(selected_idx)
            query_titles.append(name)
        else:
            print(f"⚠️ 警告: 数据集中未找到类别 {label_idx}")

    # 5. 执行搜索与绘图
    # 3行 (Query) x 6列 (1 Query + 5 Matches)
    fig, axes = plt.subplots(3, 6, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    
    for row, (q_idx, q_title) in enumerate(zip(query_indices, query_titles)):
        # 搜索
        query_feat = features_db[q_idx].unsqueeze(0)
        sim_scores = torch.mm(query_feat, features_db.t()).squeeze(0)
        topk_scores, topk_indices = torch.topk(sim_scores, k=6)
        
        # 绘制 Query (第一列)
        query_img, _ = raw_dataset[q_idx]
        axes[row, 0].imshow(query_img)
        axes[row, 0].set_title(f"QUERY\n{q_title}", color="darkred", fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # 绘制 Matches (后五列)
        for col in range(1, 6):
            idx = topk_indices[col].item()
            score = topk_scores[col].item()
            match_img, _ = raw_dataset[idx]
            
            axes[row, col].imshow(match_img)
            axes[row, col].set_title(f"Sim: {score:.3f}", fontsize=10)
            axes[row, col].axis('off')

    # 添加总标题
    plt.suptitle(f"LeJEPA Contrastive Search Demo (Model: {latest_ckpt.name})", fontsize=16, y=0.98)
    
    save_path = "demo_contrastive_result.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✅ 对比演示图已保存: {save_path}")
    print("   (请在 Windows 中打开查看，验证不同行的风格是否截然不同)")

if __name__ == "__main__":
    run_contrastive_search()
