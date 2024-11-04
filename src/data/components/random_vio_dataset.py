import torch
from torch.utils.data import Dataset
class RandomVIODataset(Dataset):
    def __init__(self,
                 seq_len=11,
                 channels=3,
                 width=512,
                 height=256,
                 imu_freq=10,
                 dataset_size=1000):
        self.seq_len = seq_len
        self.channels = channels
        self.width = width
        self.height = height
        self.imu_freq = imu_freq
        self.dataset_size = int(dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        imgs = torch.randn(self.seq_len, self.channels, self.height, self.width)
        imus = torch.randn(((self.seq_len - 1)*self.imu_freq + 1)*6)
        rot = 0.0
        weight = 0.0
        target = torch.randn(self.seq_len-1, 6)
        input = (imgs, imus, rot, weight)
        return input, target



