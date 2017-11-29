import torch
import torch.nn as nn
from torch.autograd import Variable
from distributions import DiagonalGaussian, Bernoulli
from modules.convolutional import Convolutional, ConvolutionalNetwork
from modules.latent_levels import ConvolutionalLatentLevel


class ConvDLVM(nn.Module):
    """
    Convolutional deep latent variable model.
    """
    def __init__(self, model_config):

        self.encoding_form = model_config['encoding_form']
        self.constant_variances = model_config['constant_prior_variances']
        self.concat_variables = model_config['concat_levels']
        self.output_distribution = model_config['output_distribution']

        self.levels = nn.ModuleList([])
        self.output_decoder = self.output_dist = self.mean_output = self.log_var_output = self.trainable_log_var = None
        self._construct(model_config)

    def _construct(self, model_config):

        # encode samples, gradients, errors, etc.
        encoding_form = model_config['encoding_form']

        encoder_arch = {}
        encoder_arch['non_linearity'] = model_config['non_linearity_enc']
        encoder_arch['connection_type'] = model_config['connection_type_enc']
        encoder_arch['batch_norm'] = model_config['batch_norm_enc']
        encoder_arch['weight_norm'] = model_config['weight_norm_enc']
        encoder_arch['dropout'] = model_config['dropout_enc']

        decoder_arch = {}
        decoder_arch['non_linearity'] = model_config['non_linearity_dec']
        decoder_arch['connection_type'] = model_config['connection_type_dec']
        decoder_arch['batch_norm'] = model_config['batch_norm_dec']
        decoder_arch['weight_norm'] = model_config['weight_norm_dec']
        decoder_arch['dropout'] = model_config['dropout_dec']

        # construct each level of latent variables
        for level in range(len(model_config['n_latent'])):

            # get specifications for this level's encoder and decoder
            encoder_arch['n_in'] = self.encoder_input_size(level, model_config)
            encoder_arch['n_units'] = model_config['n_units_enc'][level]
            encoder_arch['n_layers'] = model_config['n_layers_enc'][level]

            decoder_arch['n_in'] = self.decoder_input_size(level, model_config)
            decoder_arch['n_units'] = model_config['n_units_dec'][level+1]
            decoder_arch['n_layers'] = model_config['n_layers_dec'][level+1]

            n_latent = model_config['n_latent'][level]
            n_det = [model_config['n_det_enc'][level], model_config['n_det_dec'][level]]

            learn_prior = True if model_config['learn_top_prior'] else (level != len(model_config['n_latent'])-1)

            self.levels.append(DenseLatentLevel(self.batch_size, encoder_arch, decoder_arch, n_latent, n_det,
                                                  encoding_form, const_prior_var, variable_update_form, learn_prior))

        # construct the output decoder
        decoder_arch['n_in'] = self.decoder_input_size(-1, model_config)
        decoder_arch['n_units'] = model_config['n_units_dec'][0]
        decoder_arch['n_layers'] = model_config['n_layers_dec'][0]
        self.output_decoder = MultiLayerPerceptron(**decoder_arch)

        # construct the output distribution
        if self.output_distribution == 'bernoulli':
            self.output_dist = Bernoulli(None)
            self.mean_output = Dense(model_config['n_units_dec'][0], self.input_size, non_linearity='sigmoid', weight_norm=model_config['weight_norm_dec'])
        elif self.output_distribution == 'gaussian':
            self.output_dist = DiagonalGaussian(None, None)
            non_lin = 'linear' if model_config['whiten_input'] else 'sigmoid'
            self.mean_output = Dense(model_config['n_units_dec'][0], self.input_size, non_linearity=non_lin, weight_norm=model_config['weight_norm_dec'])
            if self.constant_variances:
                self.trainable_log_var = Variable(torch.zeros(self.input_size), requires_grad=True)
            else:
                self.log_var_output = Dense(model_config['n_units_dec'][0], self.input_size, weight_norm=model_config['weight_norm_dec'])

    def encoder_input_size(self, level_num, arch):
        """Calculates the size of the encoding input to a level."""

        def _encoding_size(_self, _level_num, _arch, lower_level=False):

            if _level_num == 0:
                latent_size = _self.input_size
                det_size = 0
            else:
                latent_size = _arch['n_latent'][_level_num-1]
                det_size = _arch['n_det_enc'][_level_num-1]
            encoding_size = det_size

            if 'posterior' in _self.encoding_form:
                encoding_size += latent_size
            if 'bottom_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'bottom_norm_error' in _self.encoding_form:
                encoding_size += latent_size
            if 'top_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]
            if 'top_norm_error' in _self.encoding_form and not lower_level:
                encoding_size += _arch['n_latent'][_level_num]

            return encoding_size

        encoder_size = _encoding_size(self, level_num, arch)
        if self.concat_variables:
            for level in range(level_num):
                encoder_size += _encoding_size(self, level, arch, lower_level=True)
        return encoder_size

    def decoder_input_size(self, level_num, arch):
        """Calculates the size of the decoding input to a level."""
        if level_num == len(arch['n_latent'])-1:
            return self.top_size

        decoder_size = arch['n_latent'][level_num+1] + arch['n_det_dec'][level_num+1]
        if self.concat_variables:
            for level in range(level_num+2, len(arch['n_latent'])):
                decoder_size += (arch['n_latent'][level] + arch['n_det_dec'][level])
        return decoder_size

    def process_input(self, input):
        if self.whitening_matrix is not None:
            return torch.mm(input - self.data_mean, self.whitening_matrix)
        else:
            return input / 255.

    def process_output(self, mean):
        if self.whitening_matrix is not None:
            self.reconstruction = self.data_mean + torch.mm(mean, self.inverse_whitening_matrix)
        else:
            self.reconstruction = 255. * mean

    def get_input_encoding(self, input):
        """Encoding at the bottom level."""
        if 'bottom_error' in self.encoding_form or 'bottom_norm_error' in self.encoding_form:
            assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
        encoding = None
        if 'posterior' in self.encoding_form:
            encoding = input if self.whitening_matrix is not None else input - 0.5
        if 'bottom_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach()
            encoding = error if encoding is None else torch.cat((encoding, error), dim=1)
        if 'bottom_norm_error' in self.encoding_form:
            error = input - self.output_dist.mean.detach()
            norm_error = None
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var.detach())
            elif self.output_distribution == 'bernoulli':
                mean = self.output_dist.mean.detach()
                norm_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), dim=1)
        return encoding

    def encode(self, input):
        """Encodes the input into an updated posterior estimate."""
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        input = self.process_input(input.view(-1, self.input_size))
        h = self.get_input_encoding(input)
        for latent_level in self.levels:
            if self.concat_variables:
                h = torch.cat([h, latent_level.encode(h)], dim=1)
            else:
                h = latent_level.encode(h)

    def decode(self, generate=False):
        """Decodes the posterior (prior) estimate to get a reconstruction (sample)."""
        h = Variable(torch.zeros(self.batch_size, self.top_size))
        if self._cuda_device is not None:
            h = h.cuda(self._cuda_device)
        concat = False
        for latent_level in self.levels[::-1]:
            if self.concat_variables and concat:
                h = torch.cat([h, latent_level.decode(h, generate)], dim=1)
            else:
                h = latent_level.decode(h, generate)
            concat = True

        h = self.output_decoder(h)
        self.output_dist.mean = self.mean_output(h)

        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.output_dist.log_var = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
            else:
                self.output_dist.log_var = self.log_var_output(h)

        self.process_output(self.output_dist.mean)
        return self.output_dist

    def kl_divergences(self, averaged=False):
        """Returns a list containing kl divergences at each level."""
        kl = []
        for latent_level in self.levels:
            kl.append(torch.clamp(latent_level.kl_divergence(), min=self.kl_min).sum(1))
        if averaged:
            [level_kl.mean(0) for level_kl in kl]
        else:
            return kl

    def conditional_log_likelihoods(self, input, averaged=False):
        """Returns the conditional likelihood."""
        if self._cuda_device is not None:
            input = input.cuda(self._cuda_device)
        input = self.process_input(input.view(-1, self.input_size))
        if averaged:
            return self.output_dist.log_prob(sample=input).sum(1).mean(0)
        else:
            return self.output_dist.log_prob(sample=input).sum(1)

    def elbo(self, input, averaged=False):
        """Returns the ELBO."""
        cond_like = self.conditional_log_likelihoods(input)
        kl = sum(self.kl_divergences())
        lower_bound = cond_like - kl
        if averaged:
            return lower_bound.mean(0)
        else:
            return lower_bound

    def losses(self, input, averaged=False):
        """Returns all losses."""
        cond_log_like = self.conditional_log_likelihoods(input)
        kl = self.kl_divergences()
        lower_bound = cond_log_like - sum(kl)
        if averaged:
            return lower_bound.mean(0), cond_log_like.mean(0), [level_kl.mean(0) for level_kl in kl]
        else:
            return lower_bound, cond_log_like, kl

    def reset_state(self):
        """Resets the posterior estimate."""
        for latent_level in self.levels:
            latent_level.reset()

    def trainable_state(self):
        """Makes the posterior estimate trainable."""
        for latent_level in self.levels:
            latent_level.trainable_state()

    def parameters(self):
        """Returns a list containing all parameters."""
        return self.encoder_parameters() + self.decoder_parameters()

    def encoder_parameters(self):
        """Returns a list containing all parameters in the encoder."""
        params = []
        for level in self.levels:
            params.extend(level.encoder_parameters())
        return params

    def decoder_parameters(self):
        """Returns a list containing all parameters in the decoder."""
        params = []
        for level in self.levels:
            params.extend(level.decoder_parameters())
        params.extend(list(self.output_decoder.parameters()))
        params.extend(list(self.mean_output.parameters()))
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                params.append(self.trainable_log_var)
            else:
                params.extend(list(self.log_var_output.parameters()))
        return params

    def state_parameters(self):
        """Returns a list containing the posterior estimate (state)."""
        states = []
        for latent_level in self.levels:
            states.extend(list(latent_level.state_parameters()))
        return states
