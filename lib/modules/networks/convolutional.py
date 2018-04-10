import torch
import torch.nn as nn
from lib.modules.layers import ConvolutionalLayer
from network import Network


class ConvolutionalNetwork(Network):

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
