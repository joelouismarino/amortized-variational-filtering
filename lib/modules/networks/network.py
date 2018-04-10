import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Abstract class for a neural network.
    """
    def __init__(self, network_config):
        super(Network, self).__init__()
        self.network_config = network_config

    def forward(self, input):
        """
        Abstract method for forward computation.
        """
        raise NotImplementedError

    def step(self):
        """
        Abstract method for stepping the network forward.
        """
        pass

    def re_init(self):
        """
        Abstract method to re-initialize the netwrk state.
        """
        pass
