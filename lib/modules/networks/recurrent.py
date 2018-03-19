import torch
import torch.nn as nn
from network import Network
from fully_connected import FullyConnected


class RecurrentNetwork(Network):

    def __init__(self, n_in, n_units, n_layers, connection_type='sequential', **kwargs):
        super(RecurrentNetwork, self).__init__()
        self.n_layers = n_layers
        self.connection_type = connection_type
        self.layers = nn.ModuleList([])
        self.gates = nn.ModuleList([])
        n_in_orig = n_in
        output_size = 0

        if connection_type in ['residual', 'highway']:
            self.input_map = FullyConnected(n_in, n_units)

        for _ in range(self.n_layers):
            self.layers.append(Recurrent(n_in, n_units))
            if self.connection_type == 'highway':
                self.gates.append(FullyConnected(n_in, n_units, non_linearity='sigmoid'))
            if self.connection_type in ['sequential', 'residual', 'highway']:
                n_in = n_units
            elif self.connection_type == 'concat_input':
                n_in = n_units + n_in_orig
            elif self.connection_type == 'concat':
                n_in += n_units
            output_size = n_in
        self.n_out = output_size

    def forward(self, input):
        input_orig = input.clone()
        for layer_num, layer in enumerate(self.layers):
            layer_output = layer(input)
            if self.connection_type == 'sequential':
                input = layer_output
            elif self.connection_type == 'residual':
                if layer_num == 0:
                    input = self.input_map(input) + layer_output
                else:
                    input = input + layer_output
            elif self.connection_type == 'highway':
                gate = self.gates[layer_num](input)
                if layer_num == 0:
                    input = self.input_map(input)
                input = gate * input + (1. - gate) * layer_output
            elif self.connection_type == 'concat':
                input = torch.cat((input, layer_output), dim=1)
            elif self.connection_type == 'concat_input':
                input = torch.cat((input_orig, layer_output), dim=1)
        return input

    def reset(self):
        for layer in self.layers:
            layer.reset()
