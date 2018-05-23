import torch
import torch.nn as nn
from torch.nn import init
from layer import Layer
from lib.modules.misc import ClippedLeakyReLU


class FullyConnectedLayer(Layer):
    """
    Fully-connected neural network layer.

    Args:
        layer_config (dict): dictionary containing layer configuration parameters,
                             should contain keys for n_in, n_out, non_linearity,
                             batch_norm, weight_norm, dropout, initialization
    """
    def __init__(self, layer_config):
        super(FullyConnectedLayer, self).__init__(layer_config)
        self._construct(layer_config)

    def _construct(self, layer_config):
        """
        Method to construct the layer from the layer_config dictionary parameters.
        """
        self.linear = nn.Linear(layer_config['n_in'], layer_config['n_out'])
        self.bn = lambda x: x
        if 'batch_norm' in layer_config:
            if layer_config['batch_norm']:
                self.bn = nn.BatchNorm1d(layer_config['n_out'], momentum=0.99)
        if 'weight_norm' in layer_config:
            if layer_config['weight_norm']:
                self.linear = nn.utils.weight_norm(self.linear, name='weight')

        init_gain = 1.
        if 'non_linearity' in layer_config:
            non_linearity = layer_config['non_linearity']
            if non_linearity is None or non_linearity == 'linear':
                self.non_linearity = lambda x: x
            elif non_linearity == 'relu':
                self.non_linearity = nn.ReLU()
                init_gain = init.calculate_gain('relu')
            elif non_linearity == 'leaky_relu':
                self.non_linearity = nn.LeakyReLU()
            elif non_linearity == 'clipped_leaky_relu':
                self.non_linearity = ClippedLeakyReLU(negative_slope=1./3, clip_min=-3, clip_max=3)
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
                self.dropout = nn.Dropout(layer_config['dropout'])

        if 'initialize' in layer_config:
            initialize = layer_config['initialize']
            if initialize == 'normal':
                init.normal(self.linear.weight)
            elif initialize == 'glorot_uniform':
                init.xavier_uniform(self.linear.weight, gain=init_gain)
            elif initialize == 'glorot_normal':
                init.xavier_normal(self.linear.weight, gain=init_gain)
            elif initialize == 'kaiming_uniform':
                init.kaiming_uniform(self.linear.weight)
            elif initialize == 'kaiming_normal':
                init.kaiming_normal(self.linear.weight)
            elif initialize == 'orthogonal':
                init.orthogonal(self.linear.weight, gain=init_gain)
            elif initialize == '':
                pass
            else:
                raise Exception('Parameter initialization ' + str(initialize) + ' not found.')

            if 'batch_norm' in layer_config:
                if layer_config['batch_norm']:
                    init.normal(self.bn.weight, 1, 0.02)
                    init.constant(self.bn.bias, 0.)
        else:
            init.xavier_normal(self.linear.weight, gain=init_gain)

        init.constant(self.linear.bias, 0.)

    def forward(self, input):
        """
        Method to perform forward computation.
        """
        output = self.linear(input)
        output = self.bn(output)
        output = self.non_linearity(output)
        output = self.dropout(output)
        return output
