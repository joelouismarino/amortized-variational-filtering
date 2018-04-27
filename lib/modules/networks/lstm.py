import torch
import torch.nn as nn
from network import Network
from lib.modules.layers import LSTMLayer, FullyConnectedLayer


class LSTMNetwork(Network):
    """
    LSTM neural network (with multiple layers).

    Args:
        network_config (dict): dictionary containing network configuration parameters,
                               keys should include n_in, n_units, n_layers,
                               connection_type
    """
    def __init__(self, network_config):
        super(LSTMNetwork, self).__init__(network_config)
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
        n_units = network_config['n_units']

        if self.connection_type in ['residual', 'highway']:
            # intial linear layer to embed to correct size
            self.initial_fc = FullyConnected({'n_in': n_in, 'n_out': n_units})

        for _ in range(network_config['n_layers']):
            self.layers.append(LSTMLayer({'n_in': n_in,
                                          'n_units': n_units}))
            if self.connection_type == 'highway':
                self.gates.append(FullyConnectedLayer({'n_in': n_in,
                                                       'n_out': n_units,
                                                       'non_linearity':'sigmoid'}))
            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_units
            elif self.connection_type == 'concat_input':
                n_in = n_units + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_units
            output_size = n_in

    def forward(self, input, detach=False):
        """
        Method for forward computation.
        """
        input_orig = input.clone()
        for layer_num, layer in enumerate(self.layers):
            layer_output = layer(input, detach)
            if self.connection_type == 'sequential':
                input = layer_output
            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.initial_fc(input) + layer_output
                else:
                    input = input + layer_output
            elif self.connection_type == 'highway':
                gate = self.gates[layer_num](input)
                if layer_num == 0:
                    input = self.initial_fc(input)
                input = gate * input + (1. - gate) * layer_output
            elif self.connection_type == 'concat':
                input = torch.cat((input, layer_output), dim=1)
            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer_output), dim=1)
        return input

    def re_init(self, input=None):
        """
        Method to reinitialize the hidden state and cell state within each layer.
        """
        for layer in self.layers:
            layer.re_init(input)

    def step(self):
        """
        Method to step each layer forward in the sequence.
        """
        for layer in self.layers:
            layer.step()
