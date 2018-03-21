import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Abstract class for a neural network.
    """
    def __init__(self, network_config):
        super(Network, self).__init__()

    def forward(self, input):
        """
        Abstract method for forward computation.
        """
        raise NotImplementedError
