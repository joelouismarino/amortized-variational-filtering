import torch.nn as nn
from fully_connected import FullyConnected, FullyConnectedNetwork
from convolutional import Convolutional, ConvolutionalNetwork
from variables import FullyConnectedLatentVariable, ConvolutionalLatentVariable


class FullyConnectedLatentLevel(nn.Module):
    """
    A fully-connected latent level in generalized coordinates with encoder and
    decoder networks, optional deterministic units.
    """
    def __init__(self, n_latent, n_det, n_orders_motion, encoder_arch, decoder_arch,
                  encoding_form, const_prior_var, norm_flow, learn_prior=True, dynamic=False):
        super(FullyConnectedLatentLevel, self).__init__()
        self.n_latent = n_latent
        self.n_det = n_det
        self.n_orders_motion = n_orders_motion

        self.encoder = FullyConnectedNetwork(**encoder_arch)
        self.decoder = FullyConnectedNetwork(**decoder_arch)

        var_input_sizes = (encoder_arch['n_units'], decoder_arch['n_units'])
        self.latent = FullyConnectedLatentVariable(self.n_latent, self.n_orders_motion,
                                                   const_prior_var, var_input_sizes,
                                                   learn_prior, dynamic)

        self.det_enc = FullyConnected() if self.n_det[0] > 0 else None
        self.det_dec = FullyConnected() if self.n_det[1] > 0 else None

    def infer(self, input):
        encoded = self.encoder(self.get_encoding(input, 'in'))
        output = self.get_encoding(self.latent.infer(encoded).mean(dim=1), 'out')
        if self.det_enc:
            det = self.det_enc(encoded)
            output = torch.cat((det, output), 1)
        return output

    def predict(self, input, generate):
        pass

    def get_encoding(self, input, in_out):
        pass


class ConvolutionalLatentLevel(nn.Module):

    def __init__(self):
        super(ConvolutionalLatentLevel, self).__init__()

    def infer(self, input):
        pass

    def predict(self, input, generate):
        pass
