import torch
import os
from pathlib import Path

class Config:
    # --- 项目元数据 ---
    PROJECT_NAME = "LeJEPA-Galaxy-SOTA"
    RUN_NAME = "v4-vit-small-accum-fix" 
    OUTPUT_DIR = Path("runs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    
    # --- 数据配置 ---
    DATA_PATH = "data/Galaxy10_DECals.h5"
    IMG_SIZE = 224 
    
    # --- 硬件加速 (RTX 3060 12G 适配版) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # [Fix] 物理 Batch Size 设为 32 (配合 4 Views 约占 7GB 显存)
    BATCH_SIZE = 48        
    # [Fix] 目标等效 Batch Size = 256
    EFFECTIVE_BATCH_SIZE = 240
    # 自动计算累积步数: 256 / 32 = 8
    GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    
    # [Fix] 降低 Worker 防止内存溢出
    NUM_WORKERS = 4       
    PIN_MEMORY = True
    MIXED_PRECISION = "bf16" 
    USE_COMPILE = True     
    
    # --- 模型参数 ---
    BACKBONE = "vit_small_patch16_224"
    PROJ_DIM = 256         
    HIDDEN_DIM = 2048      
    
    # --- 训练超参 ---
    EPOCHS = 500           
    LEARNING_RATE = 2e-3   
    WEIGHT_DECAY = 0.05    
    LAMBDA_REG = 0.05      
    NUM_SLICES = 2048      
    
    # Multi-View 策略
    NUM_VIEWS = 4          
    
    # --- 遥测 ---
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 50

    @classmethod
    def check_paths(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)

cfg = Config()