import torch
import torch.nn as nn
from torch.nn import init, Parameter
from fully_connected import FullyConnected


class Recurrent(nn.Module):

    def __init__(self, n_in, n_units):
        super(Recurrent, self).__init__()
        self.lstm = nn.LSTMCell(n_in, n_units)
        self.initial_hidden = Parameter(torch.zeros(1, n_units))
        self.initial_cell = Parameter(torch.zeros(1, n_units))
        self.hidden_state = None
        self.cell_state = None

    def forward(self, input):
        if self.hidden_state is None:
            self.hidden_state = self.initial_hidden.repeat(input.data.shape[0], 1)
        if self.cell_state is None:
            self.cell_state = self.initial_cell.repeat(input.data.shape[0], 1)
        self.hidden_state, self.cell_state = self.lstm.forward(input, (self.hidden_state, self.cell_state))
        return self.hidden_state

    def reset(self):
        self.hidden_state = None
        self.cell_state = None
