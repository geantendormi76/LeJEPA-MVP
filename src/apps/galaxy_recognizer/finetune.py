import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
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
    candidates = list(ckpt_dir.glob(f"*ep{target_ep}.pth"))
    if not candidates:
        print(f"⚠️ 未找到 Ep{target_ep}，尝试使用最新模型...")
        all_ckpts = sorted(list(ckpt_dir.glob("*.pth")), key=lambda x: int(re.search(r'ep(\d+)', x.name).group(1)))
        return all_ckpts[-1]
    return candidates[0]

def run_finetune():
    print("🚀 [Fine-tune] 启动全量微调模式...")
    
    # 1. 加载预训练权重
    ckpt_path = find_best_ckpt(450)
    print(f"📂 加载预训练底座: {ckpt_path.name}")
    
    device = cfg.DEVICE
    
    # 定义微调模型结构
    class LeJEPA_Classifier(nn.Module):
        def __init__(self, backbone_name, img_size, num_classes=10):
            super().__init__()
            # 加载 LeJEPA Encoder
            self.encoder = LeJEPA_Encoder(backbone_name, img_size)
            # 分类头 (Linear Probe 只是这里的一层，但现在我们要训练所有层)
            self.head = nn.Linear(self.encoder.backbone.num_features, num_classes)
            
        def forward(self, x):
            # 获取 Embedding (384维)
            embedding, _ = self.encoder(x)
            # 分类
            return self.head(embedding)

    model = LeJEPA_Classifier(cfg.BACKBONE, cfg.IMG_SIZE).to(device)
    
    # 加载权重 (过滤掉 projector，只保留 backbone)
    state_dict = torch.load(ckpt_path, map_location=device)
    clean_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("_orig_mod.", "")
        if k.startswith("backbone."):
            # 这里的 key 是 backbone.xxx，正好匹配 model.encoder.backbone.xxx
            # 但我们需要把前缀 'backbone.' 替换为 'encoder.backbone.'
            clean_state_dict[f"encoder.{k}"] = v
            
    msg = model.load_state_dict(clean_state_dict, strict=False)
    print(f"✅ 权重加载报告: {msg}")
    
    # 2. 数据准备 (标准监督学习增强)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.IMG_SIZE, scale=(0.8, 1.0)), # 微调时裁剪比例大一点
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(cfg.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    full_dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=None)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # 这里的 Dataset 需要一点 Hack，因为我们想对 Train/Val 用不同的 Transform
    # 简单起见，我们在 Collate 或者 Dataset 内部处理，或者这里直接用两个 Dataset 对象
    train_set = Galaxy10Dataset(cfg.DATA_PATH, transform=train_transform)
    val_set = Galaxy10Dataset(cfg.DATA_PATH, transform=val_transform)
    
    # 使用 indices 划分
    indices = torch.randperm(len(full_dataset)).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_loader = DataLoader(torch.utils.data.Subset(train_set, train_indices), 
                            batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(torch.utils.data.Subset(val_set, val_indices), 
                          batch_size=64, shuffle=False, num_workers=4)

    # 3. 差分学习率策略 (Differential Learning Rates) - SOTA 秘诀
    # Backbone 用小火慢炖 (防止破坏预训练特征)，Head 用大火爆炒
    optimizer = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-4}, # 预训练层: 1e-4
        {'params': model.head.parameters(), 'lr': 1e-3}     # 新层: 1e-3
    ], weight_decay=0.05)
    
    criterion = nn.CrossEntropyLoss()
    
    # 4. 训练循环 (微调只需要跑 20-30 轮)
    best_acc = 0.0
    
    for epoch in range(30):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Finetune Ep {epoch+1}/30")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                _, predicted = torch.max(logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}: Val Acc = {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_finetuned_model.pth")
            print(f"🔥 New Best! Saved.")

    print(f"✨ 微调完成！最终最佳精度: {best_acc:.2f}%")

if __name__ == "__main__":
    run_finetune()