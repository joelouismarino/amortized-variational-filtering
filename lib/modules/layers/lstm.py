import torch
import torch.nn as nn
from torch.nn import init, Parameter
from layer import Layer
import util.dtypes as dt


class LSTMLayer(Layer):
    """
    Fully-connected neural network layer.

    Args:
        layer_config (dict): dictionary containing layer configuration parameters,
                             should contain keys for n_in, n_units
    """
    # TODO: figure out smart way to run network multiple times at same time step
    #       without messing up the computational graph. may need to make a copy
    #       of self.lstm and run that network during inference
    def __init__(self, layer_config):
        super(LSTMLayer, self).__init__(layer_config)
        self._construct(layer_config)

    def _construct(self, layer_config):
        """
        Method to construct the layer from the layer_config dictionary
        """
        n_in = layer_config['n_in']
        n_units = layer_config['n_units']
        self.lstm = nn.LSTMCell(n_in, n_units)
        self.initial_hidden = Parameter(dt.zeros(1, n_units))
        self.initial_cell = Parameter(dt.zeros(1, n_units))
        self.hidden_state = self._hidden_state = None
        self.cell_state = self._cell_state = None

    def forward(self, input, detach=False):
        """
        Method to perform forward computation.
        """
        if self.hidden_state is None:
            # re-initialize the hidden state
            self.hidden_state = self.initial_hidden.repeat(input.data.shape[0], 1)
        if self.cell_state is None:
            # re-initialize the cell state
            self.cell_state = self.initial_cell.repeat(input.data.shape[0], 1)
        # detach the hidden and cell states if necessary
        hs = self.hidden_state.detach() if detach else self.hidden_state
        cs = self.cell_state.detach() if detach else self.cell_state
        # perform forward computation
        self._hidden_state, self._cell_state = self.lstm.forward(input, (hs, cs))
        return self._hidden_state

    def step(self):
        """
        Method to step the layer forward in the sequence.
        """
        self.hidden_state = self._hidden_state
        self.cell_state = self._cell_state

    def re_init(self, input=None):
        """
        Method to reinitialize the hidden state and cell state within the layer.
        """
        self._hidden_state = None
        self._cell_state = None
        if input is not None:
            self.hidden_state = self.initial_hidden.repeat(input.data.shape[0], 1)
            self.cell_state = self.initial_cell.repeat(input.data.shape[0], 1)
        else:
            self.hidden_state = None
            self.cell_state = None
