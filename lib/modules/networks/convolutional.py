import torch
import torch.nn as nn
from lib.modules.layers import ConvolutionalLayer
from network import Network


class ConvolutionalNetwork(Network):
    """
    Convolutional neural network.

    Args:
        network_config (dict): dictionary containing network configuration parameters,
                               keys should include n_in, n_filters, filter_size,
                               n_layers, non_linearity, connection_type, batch_norm,
                               weight_norm, dropout
    """
    def __init__(self, network_config):
        super(ConvolutionalNetwork, self).__init__(network_config)
        self._construct(network_config)

    def _construct(self, network_config):
        """
        Method to construct the network from the network_config dictionary parameters.
        """
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])
        if 'connection_type' in network_config:
            connection_types = ['sequential', 'residual', 'highway', 'concat_input', 'concat']
            assert network_config['connection_type'] in connection_types, 'Connection type not found.'
            self.connection_type = network_config['connection_type']
        else:
            self.connection_type = 'sequential'
        n_in = network_config['n_in']
        n_in_orig = network_config['n_in']
        n_filters = network_config['n_filters']
        filter_size = network_config['filter_size']
        batch_norm = False
        if 'batch_norm' in network_config:
            if network_config['batch_norm']:
                batch_norm = True
        weight_norm = False
        if 'weight_norm' in network_config:
            if network_config['weight_norm']:
                weight_norm = True
        non_linearity = 'linear'
        if 'non_linearity' in network_config:
            non_linearity = network_config['non_linearity']
        dropout = None
        if 'dropout' in network_config:
            dropout = network_config['dropout']
        output_size = 0

        if self.connection_type in ['residual', 'highway']:
            self.initial_conv = ConvolutionalLayer({'n_in': n_in,
                                                    'n_filters': n_filters,
                                                    'filter_size': filter_size,
                                                    'batch_norm': batch_norm,
                                                    'weight_norm': weight_norm})

        for _ in range(network_config['n_layers']):
            layer = ConvolutionalLayer({'n_in': n_in,
                                        'n_filters': n_filters,
                                        'filter_size': filter_size,
                                        'non_linearity': non_linearity,
                                        'batch_norm': batch_norm,
                                        'weight_norm': weight_norm,
                                        'dropout': dropout})
            self.layers.append(layer)

            if self.connection_type == 'highway':
                gate = ConvolutionalLayer({'n_in': n_in,
                                            'n_filters': n_filters,
                                            'filter_size': filter_size,
                                            'non_linearity': 'sigmoid',
                                            'batch_norm': batch_norm,
                                            'weight_norm': weight_norm})
                self.gates.append(gate)

            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_filters
            elif self.connection_type == 'concat_input':
                n_in = n_filters + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_filters
            output_size = n_in
        self.n_out = output_size

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
