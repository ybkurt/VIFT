import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LatentVectorDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.latent_files = os.listdir(root_dir)
    def __len__(self):
        return int(len(self.latent_files)/4)
    def __getitem__(self, idx):
        latent_vector = np.load(os.path.join(self.root_dir, f"{idx}.npy"))
        gt = np.load(os.path.join(self.root_dir, f"{idx}_gt.npy"))
        rot = np.load(os.path.join(self.root_dir, f"{idx}_rot.npy"))
        w = np.load(os.path.join(self.root_dir, f"{idx}_w.npy"))
        return (torch.from_numpy(latent_vector).to(torch.float), torch.from_numpy(rot), torch.from_numpy(w)), torch.from_numpy(gt).to(torch.float).squeeze()


