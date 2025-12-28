import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
import sys
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).resolve().parent))
if str(Path(__file__).resolve().parent / "src") not in sys.path:
     sys.path.append(str(Path(__file__).resolve().parent / "src"))

from configs.config import cfg
from src.dataset import Galaxy10Dataset, get_transforms
from src.modeling.encoder import LeJEPA_Encoder
from src.modeling.loss import SIGRegLoss
from src.telemetry import Telemetry

def train():
    # 1. 初始化
    cfg.check_paths()
    run_name = f"{cfg.RUN_NAME}_{int(time.time())}"
    telemetry = Telemetry(log_dir=cfg.OUTPUT_DIR / run_name)
    
    telemetry.info(f"🚀 [SOTA-Final] 训练启动: {run_name}")
    telemetry.info(f"⚙️ [Config] Physical Batch: {cfg.BATCH_SIZE} | Accum Steps: {cfg.GRAD_ACCUM_STEPS} | Effective Batch: {cfg.EFFECTIVE_BATCH_SIZE}")
    telemetry.info(f"⚙️ [Config] Views: {cfg.NUM_VIEWS} | Epochs: {cfg.EPOCHS}")

    # 2. 数据加载
    dataset = Galaxy10Dataset(cfg.DATA_PATH, transform=get_transforms(cfg.IMG_SIZE, cfg.NUM_VIEWS))
    loader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
        persistent_workers=True
    )

    # 3. 模型与损失
    model = LeJEPA_Encoder(
        backbone_name=cfg.BACKBONE,
        img_size=cfg.IMG_SIZE,
        proj_dim=cfg.PROJ_DIM
    ).to(cfg.DEVICE)
    
    if cfg.USE_COMPILE:
        print("⚡启用 torch.compile 加速...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"⚠️ torch.compile 失败，回退到 eager 模式: {e}")
    
    sigreg_criterion = SIGRegLoss(num_slices=cfg.NUM_SLICES, t_max=5.0).to(cfg.DEVICE)
    
    # 4. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # Scheduler
    total_batches = len(loader) * cfg.EPOCHS
    total_updates = total_batches // cfg.GRAD_ACCUM_STEPS
    warmup_updates = int(0.05 * total_updates)
    
    scheduler1 = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_updates)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_updates - warmup_updates, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_updates])
    
    scaler = GradScaler('cuda')

    # 5. 训练循环
    global_step = 0
    model.train()
    optimizer.zero_grad()

    for epoch in range(cfg.EPOCHS):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Ep {epoch+1}/{cfg.EPOCHS}")
        
        # [Fix] 显式解包: (images, labels)
        # images shape: [B, V, C, H, W]
        for batch_idx, (images, _) in enumerate(pbar):
            
            images = images.to(cfg.DEVICE, non_blocking=True)
            B, V = images.shape[:2]
            
            # [Fix] Flatten: [B, V, C, H, W] -> [B*V, C, H, W]
            # 这种写法比 torch.cat 更安全，且零拷贝
            all_imgs = images.flatten(0, 1)
            
            with autocast('cuda', dtype=torch.bfloat16):
                _, projections = model(all_imgs)
                
                # Reshape back: [B*V, D] -> [B, V, D] -> [V, B, D]
                # LeJEPA Loss 需要 Views 在第一维
                z = projections.view(B, V, -1).transpose(0, 1)
                
                # Loss 计算
                z_mean = z.mean(dim=0) # [B, D]
                pred_loss = (z - z_mean.unsqueeze(0)).pow(2).mean()
                
                # SIGReg
                sigreg_val = sigreg_criterion(projections)
                
                loss = (1 - cfg.LAMBDA_REG) * pred_loss + cfg.LAMBDA_REG * sigreg_val
                loss = loss / cfg.GRAD_ACCUM_STEPS
            
            scaler.scale(loss).backward()
            
            current_loss = loss.item() * cfg.GRAD_ACCUM_STEPS
            epoch_loss += current_loss

            if (batch_idx + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                if global_step % cfg.LOG_INTERVAL == 0:
                    telemetry.log_metrics({
                        "Loss/Total": current_loss,
                        "Loss/Pred": pred_loss.item(),
                        "Loss/SIGReg": sigreg_val.item(),
                        "LR": optimizer.param_groups[0]['lr']
                    }, global_step)
            
            pbar.set_postfix({"L": f"{current_loss:.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.1e}"})

        avg_loss = epoch_loss / len(loader)
        telemetry.log_metrics({"Loss/Epoch": avg_loss}, epoch)
        
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            save_path = cfg.CHECKPOINT_DIR / f"{run_name}_ep{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            telemetry.info(f"💾 模型已保存: {save_path.name}")

    telemetry.close()
    print("✨ SOTA 级训练完成！")

if __name__ == "__main__":
    train()