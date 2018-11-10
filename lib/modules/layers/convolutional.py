import torch
import torch.nn as nn
from torch.nn import init
from layer import Layer


class ConvolutionalLayer(Layer):
    """
    Convolutional neural network layer.

    Args:
        layer_config (dict): dictionary containing layer configuration parameters,
                             should contain keys for n_in, n_out, filter_size,
                             non_linearity, batch_norm, weight_norm, dropout,
                             initialize
    """
    def __init__(self, layer_config):
        super(ConvolutionalLayer, self).__init__(layer_config)
        self._construct(layer_config)

    def _construct(self, layer_config):

        n_in = layer_config['n_in']
        n_out = layer_config['n_filters']
        filter_size = layer_config['filter_size']
        stride=1
        if 'stride' in layer_config:
            if layer_config['stride']:
                stride = layer_config['stride']
        self.conv = nn.Conv2d(n_in, n_out, filter_size, padding=int(filter_size)/2, stride=stride)
        self.bn = lambda x: x
        if 'batch_norm' in layer_config:
            if layer_config['batch_norm']:
                self.bn = nn.BatchNorm2d(n_out)
        if 'weight_norm' in layer_config:
            if layer_config['weight_norm']:
                self.conv = nn.utils.weight_norm(self.conv, name='weight')

        init_gain = 1.
        if 'non_linearity' in layer_config:
            non_linearity = layer_config['non_linearity']
            if non_linearity is None:
                self.non_linearity = None
            elif non_linearity == 'relu':
                self.non_linearity = nn.ReLU()
                init_gain = init.calculate_gain('relu')
            elif non_linearity == 'elu':
                self.non_linearity = nn.ELU()
            elif non_linearity == 'selu':
                self.non_linearity = nn.SELU()
            elif non_linearity == 'tanh':
                self.non_linearity = nn.Tanh()
                init_gain = init.calculate_gain('tanh')
            elif non_linearity == 'sigmoid':
                self.non_linearity = nn.Sigmoid()
            else:
                raise Exception('Non-linearity ' + str(non_linearity) + ' not found.')
        else:
            self.non_linearity = lambda x: x

        self.dropout = lambda x: x
        if 'dropout' in layer_config:
            if layer_config['dropout'] is not None:
                self.dropout = nn.Dropout2d(layer_config['dropout'])

        if 'initialize' in layer_config:
            initialize = layer_config['initialize']
            if initialize == 'normal':
                init.normal(self.conv.weight)
            elif initialize == 'glorot_uniform':
                init.xavier_uniform(self.conv.weight, gain=init_gain)
            elif initialize == 'glorot_normal':
                init.xavier_normal(self.conv.weight, gain=init_gain)
            elif initialize == 'kaiming_uniform':
                init.kaiming_uniform(self.conv.weight)
            elif initialize == 'kaiming_normal':
                init.kaiming_normal(self.conv.weight)
            elif initialize == 'orthogonal':
                init.orthogonal(self.conv.weight, gain=init_gain)
            elif initialize == '':
                pass
            else:
                raise Exception('Parameter initialization ' + str(initialize) + ' not found.')
        else:
            init.xavier_normal(self.conv.weight, gain=init_gain)

        if 'batch_norm' in layer_config:
            if layer_config['batch_norm']:
                init.constant(self.bn.weight, 1.)
                init.constant(self.bn.bias, 0.)

        init.constant(self.conv.bias, 0.)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.non_linearity(output)
        output = self.dropout(output)
        return output
