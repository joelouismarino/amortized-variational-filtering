import torch
import torch.nn as nn
from fully_connected import FullyConnected
from convolutional import Convolutional


class FullyConnectedIAF(nn.Module):

    def __init__(self, n_variables):
        super(FullyConnectedIAF, self).__init__()
        self.mean = FullyConnected(n_variables, n_variables)
        self.std = FullyConnected(n_variables, n_variables)

    def forward(self, input):
        return


class ConvolutionalIAF(nn.Module):

    def __init__(self, n_variables, filter_size):
        super(ConvolutionalIAF, self).__init__()
        self.mean = Convolutional(n_variables, n_variables, filter_size)
        self.std = Convolutional(n_variables, n_variables, filter_size)

    def forward(self, input):
        return
