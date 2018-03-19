import torch
import torch.nn as nn
from torch.nn import init


class FullyConnected(nn.Module):
    """
    Fully-connected (dense) layer with optional batch normalization,
    non-linearity, weight normalization, and dropout.
    """
    def __init__(self, n_in, n_out, non_linearity=None, batch_norm=False,
                weight_norm=False, dropout=0., initialize='glorot_uniform'):
        super(FullyConnected, self).__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm1d(n_out, momentum=0.99)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear, name='weight')

        init_gain = 1.

        if non_linearity is None or non_linearity == 'linear':
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

        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)

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

        if batch_norm:
            init.normal(self.bn.weight, 1, 0.02)
            init.constant(self.bn.bias, 0.)

        init.constant(self.linear.bias, 0.)

    def forward(self, input):
        output = self.linear(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output
