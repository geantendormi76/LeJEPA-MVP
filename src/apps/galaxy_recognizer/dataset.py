import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MultiViewTransform:
    """
    [SOTA Aligned] 生成 N 个全局视图，并堆叠为 Tensor。
    Output Shape: [V, C, H, W]
    """
    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        # [Critical] 返回 Tensor 而非 List
        return torch.stack([self.base_transform(x) for _ in range(self.n_views)])

class Galaxy10Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.transform = transform
        print(f"📂 [Dataset] 加载数据: {h5_path} ...")
        with h5py.File(h5_path, 'r') as F:
            self.images = F['images'][:] 
            self.labels = F['ans'][:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.fromarray(self.images[idx])
        if self.transform:
            return self.transform(img), self.labels[idx]
        return img, self.labels[idx]

def get_transforms(img_size=224, n_views=2):
    train_aug = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return MultiViewTransform(train_aug, n_views=n_views)