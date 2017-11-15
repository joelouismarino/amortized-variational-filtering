import torch.nn as nn
from fully_connected import FullyConnectedNetwork
from convolutional import ConvolutionalNetwork
from variables import FullyConnectedLatentVariable, ConvolutionalLatentVariable


class FullyConnectedLatentLevel(nn.Module):

    def __init__(self):
        super(FullyConnectedLatentLevel, self).__init__()

    def infer(self, input):
        pass

    def predict(self, input, generate):
        pass

    def forward(self, input, mode):
        if mode == 'infer':
            return self.infer(input)
        elif mode == 'predict':
            return self.predict(input, generate=False)
        elif mode == 'generate':
            return self.predict(input, generate=True)


class ConvolutionalLatentLevel(nn.Module):

    def __init__(self):
        super(ConvolutionalLatentLevel, self).__init__()

    def infer(self, input):
        pass

    def predict(self, input, generate):
        pass

    def forward(self, input, mode):
        if mode == 'infer':
            return self.infer(input)
        elif mode == 'predict':
            return self.predict(input, generate=False)
        elif mode == 'generate':
            return self.predict(input, generate=True)
