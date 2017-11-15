import torch
import torch.nn as nn
from torch.nn import init


class Convolutional(nn.Module):
    """
    Basic convolutional layer with optional batch normalization,
    non-linearity, weight normalization and dropout.
    """
    def __init__(self, n_in, n_out, filter_size, non_linearity=None,
                batch_norm=False, weight_norm=False, dropout=0., initialize='glorot_uniform'):
        super(Convolutional, self).__init__()

        self.conv = nn.Conv2d(n_in, n_out, filter_size, padding=int(filter_size)/2)
        self.bn = None
        if batch_norm:
            self.bn = nn.BatchNorm2d(n_out)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')

        init_gain = 1.

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

        self.dropout = None
        if dropout > 0.:
            self.dropout = nn.Dropout2d(dropout)

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

        if batch_norm:
            init.constant(self.bn.weight, 1.)
            init.constant(self.bn.bias, 0.)

        init.constant(self.conv.bias, 0.)

    def forward(self, input):
        output = self.conv(input)
        if self.bn:
            output = self.bn(output)
        if self.non_linearity:
            output = self.non_linearity(output)
        if self.dropout:
            output = self.dropout(output)
        return output


class ConvolutionalNetwork(nn.Module):

    """Multi-layer convolutional network."""

    def __init__(self, n_in, n_filters, filter_size, n_layers, non_linearity=None,
                connection_type='sequential', batch_norm=False, weight_norm=False, dropout=0.):
        super(ConvolutionalNetwork, self).__init__()
        assert connection_type in ['sequential', 'residual', 'highway', 'concat_input', 'concat'], 'Connection type not found.'
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])

        n_in_orig = n_in

        if self.connection_type in ['residual', 'highway']:
            self.initial_conv = Convolutional(n_in, n_filters, filter_size, batch_norm=batch_norm, weight_norm=weight_norm)

        for _ in range(n_layers):
            layer = Convolutional(n_in, n_filters, filter_size, non_linearity=non_linearity, batch_norm=batch_norm, weight_norm=weight_norm, dropout=dropout)
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = Convolutional(n_in, n_filters, filter_size, non_linearity='sigmoid', batch_norm=batch_norm, weight_norm=weight_norm)
                self.gates.append(gate)

            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_filters
            elif self.connection_type == 'concat_input':
                n_in = n_filters + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_filters

    def forward(self, input):

        input_orig = input.clone()

        for layer_num, layer in enumerate(self.layers):
            if self.connection_type == 'sequential':
                input = layer(input)

            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_conv(input) + layer(input)
                else:
                    input += layer(input)

            elif self.connection_type == 'highway':
                gate = self.gates[layer_num]
                if layer_num == 0:
                    input = gate(input) * self.initial_conv(input) + (1 - gate(input)) * layer(input)
                else:
                    input = gate(input) * input + (1 - gate(input)) * layer(input)

            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer(input)), dim=1)

            elif self.connection_type == 'concat':
                input = torch.cat((input, layer(input)), dim=1)

        return input
