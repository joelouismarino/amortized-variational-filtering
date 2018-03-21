import torch
from latent_level import LatentLevel
from lib.modules.layers import FullyConnectedLayer
from lib.modules.networks import FullyConnectedNetwork
from lib.modules.latent_variables import FullyConnectedLatentVariable


class FullyConnectedLatentLevel(LatentLevel):
    """
    A fully-connected latent level with encoder and decoder networks,
    optional deterministic connections.
    """
    def __init__(self, n_latent, n_det, encoder_arch, decoder_arch, encoding_form,
                 const_prior_var, norm_flow):
        super(FullyConnectedLatentLevel, self).__init__()
        self.n_latent = n_latent
        self.n_det = n_det
        self.encoding_form = encoding_form

        self.encoder = FullyConnectedNetwork(**encoder_arch)
        self.decoder = FullyConnectedNetwork(**decoder_arch)

        var_input_sizes = (encoder_arch['n_units'], decoder_arch['n_units'])
        self.latent = FullyConnectedLatentVariable(self.n_latent, const_prior_var,
                                                   var_input_sizes, norm_flow)

        self.det_enc = FullyConnected(encoder_arch['n_units'], self.n_det) if self.n_det[0] > 0 else None
        self.det_dec = FullyConnected(decoder_arch['n_units'], self.n_det) if self.n_det[1] > 0 else None

    def infer(self, input):
        encoded = self.encoder(self._get_encoding_form(input, 'in'))
        output = self._get_encoding_form(self.latent.infer(encoded).mean(dim=1), 'out')
        if self.det_enc:
            det = self.det_enc(encoded)
            output = torch.cat((det, output), 1)
        return output

    def generate(self, input, n_samples, gen):
        b, s, n = input.data.shape
        decoded = self.decoder(input.view(-1, n)).view(b, s, -1)
        sample = self.latent.predict(decoded, n_samples, gen)
        if self.det_dec:
            n = decoded.data.shape[2]
            det = self.det_dec(decoded.view(-1, n)).view(b, n_samples, -1)
            sample = torch.cat((sample, det), dim=2)
        return sample

    def _get_encoding_form(self, input, in_out):
        pass
