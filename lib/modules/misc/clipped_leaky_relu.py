import torch
import torch.nn as nn


class ClippedLeakyReLU(nn.Module):

    def __init__(self, negative_slope=0.01, clip_min=None, clip_max=None, inplace=False):
        super(ClippedLeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope, inplace)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, input):
        input = self.leaky_relu(input)
        if self.clip_min:
            input = torch.clamp(input, min=self.clip_min)
        if self.clip_max:
            input = torch.clamp(input, max=self.clip_max)
        return input
