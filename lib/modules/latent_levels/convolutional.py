import torch
import torch.nn as nn
from latent_level import LatentLevel
from convolutional import Convolutional, ConvolutionalNetwork
from variables import ConvolutionalLatentVariable


class ConvolutionalLatentLevel(LatentLevel):
    """
    A convolutional latent level with encoder and decoder networks,
    optional deterministic connections.
    """
    def __init__(self, n_variable_channels, filter_size, n_det, encoder_arch,
                decoder_arch, encoding_form, const_prior_var, norm_flow):
        super(ConvolutionalLatentLevel, self).__init__()
        self.n_variable_channels = n_variable_channels
        self.n_det = n_det
        self.encoding_form = encoding_form

        self.encoder = ConvolutionalNetwork(**encoder_arch)
        self.decoder = ConvolutionalNetwork(**decoder_arch)

        var_input_sizes = (encoder_arch['n_filters'], decoder_arch['n_filters'])
        self.latent = ConvolutionalLatentVariable(self.n_variable_channels, filter_size,
                                                  const_prior_var, var_input_sizes,
                                                  norm_flow)

        self.det_enc = Convolutional(encoder_arch['n_filters'], self.n_det, filter_size) if self.n_det[0] > 0 else None
        self.det_dec = Convolutional(decoder_arch['n_filters'], self.n_det, filter_size) if self.n_det[1] > 0 else None

    def infer(self, input):
        encoded = self.encoder(self._get_encoding_form(input, 'in'))
        output = self._get_encoding_form(self.latent.infer(encoded).mean(dim=1), 'out')
        if self.det_enc:
            det = self.det_enc(encoded)
            output = torch.cat((det, output), 1)
        return output

    def generate(self, input, gen, n_samples):
        input = self._get_decoding_form(input, n_samples)
        b, s, c, h, w = input.data.shape
        decoded = self.decoder(input.view(-1, c, h, w)).view(b, s, -1, h, w)
        sample = self.latent.generate(decoded, gen, n_samples)
        if self.det_dec:
            c = decoded.data.shape[2]
            det = self.det_dec(decoded.view(-1, c, h, w)).view(b, n_samples, -1, h, w)
            sample = torch.cat((sample, det), dim=2)
        return sample

    def _get_encoding_form(self, input, in_out):
        encoding = input if in_out == 'in' else None
        if 'posterior' in self.encoding_form and in_out == 'out':
            encoding = input
        if ('top_error' in self.encoding_form and in_out == 'in') or ('bottom_error' in self.encoding_form and in_out == 'out'):
            error = self.latent.error()
            encoding = error if encoding is None else torch.cat((encoding, error), 1)
        if 'layer_norm_error' in self.encoding_form:
            weighted_error = self.latent.error(weighted=True)
            b, c, h, w = weighted_error.data.shape
            weighted_error = weighted_error.view(b, c, -1)
            weighted_error_mean = weighted_error.mean(dim=2, keepdim=True).view(b, c, 1, 1).repeat(1, 1, h, w)
            weighted_error_std = weighted_error.std(dim=2, keepdim=True).view(b, c, 1, 1).repeat(1, 1, h, w)
            norm_weighted_error = (weighted_error.view(b, c, h, w) - weighted_error_mean) / (weighted_error_std + 1e-6)
            encoding = norm_weighted_error if encoding is None else torch.cat((encoding, norm_weighted_error), 1)
        if ('top_weighted_error' in self.encoding_form and in_out == 'in') or ('bottom_weighted_error' in self.encoding_form and in_out == 'out'):
            weighted_error = self.latent.error(weighted=True)
            encoding = weighted_error if encoding is None else torch.cat((encoding, weighted_error), 1)
        if 'mean' in self.encoding_form and in_out == 'in':
            approx_post_mean = self.latent.posterior.mean.detach()
            if len(approx_post_mean.data.shape) in [3, 5]:
                approx_post_mean = approx_post_mean.mean(dim=1)
            encoding = approx_post_mean if encoding is None else torch.cat((encoding, approx_post_mean), 1)
        if 'log_var' in self.encoding_form and in_out == 'in':
            approx_post_log_var = self.latent.posterior.log_var.detach()
            if len(approx_post_log_var.data.shape) in [3, 5]:
                approx_post_log_var = approx_post_mean.mean(dim=1)
            encoding = approx_post_log_var if encoding is None else torch.cat((encoding, approx_post_log_var), 1)
        if 'mean_gradient' in self.encoding_form and in_out == 'in':
            encoding = self.latent.approx_posterior_gradients()[0] if encoding is None else torch.cat((encoding, self.latent.approx_posterior_gradients()[0]), 1)
        if 'log_var_gradient' in self.encoding_form and in_out == 'in':
            encoding = self.latent.approx_posterior_gradients()[1] if encoding is None else torch.cat((encoding, self.latent.approx_posterior_gradients()[1]), 1)
        if 'layer_norm_gradient' in self.encoding_form and in_out == 'in':
            # TODO: this is averaging over the wrong dimensions
            mean_grad, log_var_grad = self.state_gradients()
            mean_grad_mean = mean_grad.mean(dim=0, keepdim=True)
            mean_grad_std = mean_grad.std(dim=0, keepdim=True)
            norm_mean_grad = (mean_grad - mean_grad_mean) / (mean_grad_std + 1e-6)
            log_var_grad_mean = log_var_grad.mean(dim=0, keepdim=True)
            log_var_grad_std = log_var_grad.std(dim=0, keepdim=True)
            norm_log_var_grad = (log_var_grad - log_var_grad_mean) / (log_var_grad_std + 1e-6)
            norm_grads = [norm_mean_grad, norm_log_var_grad]
            encoding = norm_grads if encoding is None else torch.cat((encoding, norm_grads), 1)
        return encoding

    def _get_decoding_form(self, input, n_samples):
        prev = self.latent.previous_posterior.sample(n_samples)
        if input is None:
            # no top-down input
            return prev
        return torch.cat((prev, input), dim=2)
