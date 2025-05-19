"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(torch.nn.Module):
    def __init__(self,output_size,in_channels):
        super(FeedForwardNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels,out_channels=2,kernel_size=3)
        self.conv2 = torch.nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3)
        self.flatten1 = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(49, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output_size)
        self.output_size = output_size

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = self.conv1(x)
        x = torch.nn.Tanh()(self.flatten1(self.conv2(x)))
        x = torch.nn.Tanh()(self.fc1(x))
        x = torch.nn.Tanh()(self.fc2(x))
        if self.output_size == 1:
            return torch.nn.Tanh()(self.fc3(x))
        return torch.nn.Softmax(dim=1)(self.fc3(x))
