import torch
import torch.nn as nn

class SIGRegLoss(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization (SIGReg) Loss.
    [Upgrade] 参数对齐论文: t_max=5.0, knots=20
    """
    def __init__(self, num_slices=2048, t_max=5.0, knots=20):
        super().__init__()
        self.num_slices = num_slices
        
        # 积分网格 [0, t_max] (利用对称性)
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt 
        
        # 目标特征函数 (Standard Gaussian CF: exp(-t^2/2))
        window = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings):
        # embeddings: [B, D]
        B, D = embeddings.shape
        
        # 1. 随机投影 (Slicing)
        # A: [D, K]
        A = torch.randn(D, self.num_slices, device=embeddings.device)
        A = A.div_(A.norm(p=2, dim=0))

        # projections: [B, K]
        projections = embeddings @ A
        
        # 2. 计算 ECF (Empirical Characteristic Function)
        # x_t: [B, K, T]
        x_t = projections.unsqueeze(-1) * self.t
        
        # 对 Batch 维度求平均 -> [K, T]
        ecf_real = x_t.cos().mean(dim=0)
        ecf_imag = x_t.sin().mean(dim=0)
        
        # 3. 加权 L2 距离 (Epps-Pulley Test)
        # err: [K, T]
        err = (ecf_real - self.phi).square() + ecf_imag.square()
        
        # 对 T 维度积分 -> [K]
        statistic = (err @ self.weights) * D
        
        # 对 K (Slices) 维度求平均
        return statistic.mean()