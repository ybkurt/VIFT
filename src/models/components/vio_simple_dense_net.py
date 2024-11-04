import torch
from torch import nn


class VIOSimpleDenseNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        seq_len: int = 11,
        channels: int = 3,
        width: int = 512,
        height: int = 256,
        imu_freq: int = 10,
        lin1_size: int = 256,
        lin2_size: int = 256,
        lin3_size: int = 256,
        output_size: int = 6,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        
        input_size = (seq_len) * channels * width * height + ((seq_len - 1) * imu_freq  + 1)* 6
        output_size = (seq_len-1) * 6
        self.model = nn.Sequential(
            nn.Linear(input_size, lin1_size),
            nn.BatchNorm1d(lin1_size),
            nn.ReLU(),
            nn.Linear(lin1_size, lin2_size),
            nn.BatchNorm1d(lin2_size),
            nn.ReLU(),
            nn.Linear(lin2_size, lin3_size),
            nn.BatchNorm1d(lin3_size),
            nn.ReLU(),
            nn.Linear(lin3_size, output_size),
        )

    def forward(self, input, target):

        """Perform a single forward pass through the network.

        :param input: The input tensor.
        :param target: The target tensor.
        :return: A tensor of predictions.
        """

        imgs, imus, rot, weight = input
        batch_size, seq_len, channels, height, width = imgs.shape
        imgs = imgs.view(batch_size, -1)
        imus = imus.view(batch_size, -1)
        
        x = torch.cat((imgs,imus),1)
        return self.model(x).view(batch_size, seq_len-1, 6)

if __name__ == "__main__":
    _ = VIOSimpleDenseNet()
