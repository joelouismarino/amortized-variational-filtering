import torch.nn as nn
from fully_connected import FullyConnected, FullyConnectedNetwork
from convolutional import Convolutional, ConvolutionalNetwork


class FullyConnectedLatentVariable(nn.Module):

    def __init__(self, batch_size, n_variables, n_orders_motion, const_prior_var,
                n_input, posterior_form='gaussian', learn_prior=True, dynamic=False):
        super(FullyConnectedLatentVariable, self).__init__()

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


class ConvolutionalLatentVariable(nn.Module):

    def __init__(self, batch_size, n_variable_channels, filter_size, n_orders_motion,
                const_prior_var, n_input, posterior_form='gaussian', learn_prior=True, dynamic=False):
        super(ConvolutionalLatentVariable, self).__init__()

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
