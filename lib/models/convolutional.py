import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Normal, Bernoulli
from lib.modules.layers import ConvLayer
from lib.modules.networks import ConvNetwork
from lib.modules.latent_levels import ConvLatentLevel


class ConvDLVM(nn.Module):
    """
    Convolutional dynamical latent variable model.
    """
    def __init__(self, model_config):
        super(ConvDLVM, self).__init__()

        self.encoding_form = model_config['encoding_form']
        self.transform_input = model_config['transform_input']
        self.constant_variances = model_config['constant_prior_variances']
        self.concat_variables = model_config['concat_levels']
        self.output_distribution = model_config['output_distribution']

        self.levels = nn.ModuleList([])
        self.output_decoder = self.output_dist = self.mean_output = self.log_var_output = self.trainable_log_var = None
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Constructs the convolutional dynamical latent variable model using the model
        configuration file.
        """
        # transformation of variables on the input
        if model_config['transform_input']:
            pass

        # global model configuration parameters
        encoding_form = model_config['encoding_form']
        norm_flow = model_config['normalizing_flows']

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

            # get configuration parameters for this level
            encoder_arch['n_in'] = self._encoder_input_size(level, model_config)
            encoder_arch['n_filters'] = model_config['n_filters_enc'][level]
            encoder_arch['n_layers'] = model_config['n_layers_enc'][level]
            encoder_arch['filter_size'] = model_config['filter_size'][level]

            decoder_arch['n_in'] = self._decoder_input_size(level, model_config)
            decoder_arch['n_filters'] = model_config['n_filters_dec'][level+1]
            decoder_arch['n_layers'] = model_config['n_layers_dec'][level+1]
            decoder_arch['filter_size'] = model_config['filter_size'][level]

            filter_size = model_config['filter_size'][level]

            n_latent = model_config['n_latent'][level]
            n_det = [model_config['n_det_enc'][level], model_config['n_det_dec'][level]]

            self.levels.append(ConvolutionalLatentLevel(n_latent, filter_size, n_det, encoder_arch,
                                                        decoder_arch, encoding_form, self.constant_variances, norm_flow))

        # construct the output decoder
        decoder_arch['n_in'] = self._decoder_input_size(-1, model_config)
        decoder_arch['n_filters'] = model_config['n_filters_dec'][0]
        decoder_arch['n_layers'] = model_config['n_layers_dec'][0]
        decoder_arch['filter_size'] = model_config['filter_size'][0]
        self.output_decoder = ConvolutionalNetwork(**decoder_arch)

        # construct the output distribution
        if self.output_distribution == 'bernoulli':
            self.output_dist = Bernoulli(None)
            self.mean_output = Convolutional(model_config['n_filters_dec'][0], 3, model_config['filter_size'][0], non_linearity='sigmoid', weight_norm=model_config['weight_norm_dec'])
        elif self.output_distribution == 'gaussian':
            self.output_dist = DiagonalGaussian(None, None)
            self.mean_output = Convolutional(model_config['n_filters_dec'][0], 3, model_config['filter_size'][0], non_linearity='sigmoid', weight_norm=model_config['weight_norm_dec'])
            if self.constant_variances:
                self.trainable_log_var = Variable(torch.zeros(self.input_size), requires_grad=True)
            else:
                self.log_var_output = Convolutional(model_config['n_filters_dec'][0], 3, model_config['filter_size'][0], weight_norm=model_config['weight_norm_dec'])

    def _encoder_input_size(self, level, model_config):
        """
        Calculates the number of input filters for the encoder at a level.
        """

        def _encoding_size(_self, _level, _model_config, lower_level=False):

            top_size = _model_config['n_latent'][_level]
            if _level == 0:
                bottom_size = 3 # data channels
                det_size = 0
            else:
                bottom_size = _model_config['n_latent'][_level-1]
                det_size = _model_config['n_det_enc'][_level-1]
            encoding_size = det_size

            if 'posterior' in _self.encoding_form:
                encoding_size += bottom_size
            if 'mean' in _self.encoding_form:
                encoding_size += top_size
            if 'log_var' in _self.encoding_form:
                encoding_size += top_size
            if 'mean_gradient' in _self.encoding_form:
                encoding_size += top_size
            if 'log_var_gradient' in _self.encoding_form:
                encoding_size += top_size
            if 'bottom_error' in _self.encoding_form:
                encoding_size += bottom_size
            if 'bottom_weighted_error' in _self.encoding_form:
                encoding_size += bottom_size
            if 'top_error' in _self.encoding_form and not lower_level:
                encoding_size += top_size
            if 'top_weighted_error' in _self.encoding_form and not lower_level:
                encoding_size += top_size
            if 'layer_norm_error' in _self.encoding_form and not lower_level:
                encoding_size += (bottom_size + top_size)

            return encoding_size

        encoder_size = _encoding_size(self, level, model_config)
        if self.concat_variables:
            for level in range(level):
                encoder_size += _encoding_size(self, level, model_config, lower_level=True)
        return encoder_size

    def _decoder_input_size(self, level, model_config):
        """Calculates the size of the decoding input to a level."""
        if level == -1:
            return model_config['n_latent'][level+1] + model_config['n_det_dec'][level+1]
        decoder_size = model_config['n_latent'][level] # from previous time step
        if level < len(model_config['n_latent'])-1:
            # from higher level variable
            decoder_size += model_config['n_latent'][level+1] + model_config['n_det_dec'][level+1]
        if self.concat_variables and (level+2) < len(model_config['n_latent']):
            for l in range(level+2, len(model_config['n_latent'])):
                decoder_size += (model_config['n_latent'][l] + model_config['n_det_dec'][l])
        return decoder_size

    def _get_encoding_form(self, input):
        """Encoding at the bottom level."""
        if 'bottom_error' in self.encoding_form or 'bottom_weighted_error' in self.encoding_form:
            assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
        encoding = None
        if 'posterior' in self.encoding_form:
            encoding = input
        if 'bottom_error' in self.encoding_form:
            n_samples = self.output_dist.mean.detach().data.shape[1]
            error = (input.unsqueeze(1).repeat(1, n_samples, 1, 1, 1) - self.output_dist.mean.detach()).mean(dim=1)
            encoding = error if encoding is None else torch.cat((encoding, error), dim=1)
        if 'bottom_weighted_error' in self.encoding_form:
            n_samples = self.output_dist.mean.detach().data.shape[1]
            error = input.unsqueeze(1).repeat(1, n_samples, 1, 1, 1) - self.output_dist.mean.detach()
            norm_error = None
            if self.output_distribution == 'gaussian':
                norm_error = error / torch.exp(self.output_dist.log_var.detach())
            elif self.output_distribution == 'bernoulli':
                mean = self.output_dist.mean.detach()
                norm_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
            norm_error = norm_error.mean(dim=1)
            encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), dim=1)
        if 'layer_norm_error' in self.encoding_form:
            n_samples = self.output_dist.mean.detach().data.shape[1]
            error = input.unsqueeze(1).repeat(1, n_samples, 1, 1, 1) - self.output_dist.mean.detach()
            weighted_error = None
            if self.output_distribution == 'gaussian':
                weighted_error = error / torch.exp(self.output_dist.log_var.detach())
            elif self.output_distribution == 'bernoulli':
                mean = self.output_dist.mean.detach()
                weighted_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
            weighted_error = weighted_error.mean(dim=1)
            b, c, h, w = weighted_error.data.shape
            weighted_error = weighted_error.view(b, c, -1)
            weighted_error_mean = weighted_error.mean(dim=2, keepdim=True).view(b, c, 1, 1).repeat(1, 1, h, w)
            weighted_error_std = weighted_error.std(dim=2, keepdim=True).view(b, c, 1, 1).repeat(1, 1, h, w)
            norm_weighted_error = (weighted_error.view(b, c, h, w) - weighted_error_mean) / (weighted_error_std + 1e-6)
            encoding = norm_weighted_error if encoding is None else torch.cat((encoding, norm_weighted_error), dim=1)
        return encoding

    def infer(self, input):
        """
        Infer the approximate posterior of the latent variables.
        """
        if self.transform_input:
            pass
        h = self._get_encoding_form(input)
        for latent_level in self.levels:
            if self.concat_variables:
                h = torch.cat([h, latent_level.infer(h)], dim=1)
            else:
                h = latent_level.infer(h)

    def generate(self, gen=False, n_samples=1):
        """
        Generate observations by running the generative model forward.
        """
        hidden = None
        concat = False
        for latent_level in list(self.levels)[::-1]:
            if self.concat_variables and concat:
                hidden = torch.cat([h, latent_level.generate(hidden, gen, n_samples)], dim=2)
            else:
                hidden = latent_level.generate(hidden, gen, n_samples)
            concat = True
        b, s, c, h, w = hidden.data.shape
        hidden = self.output_decoder(hidden.view(-1, c, h, w))
        self.output_dist.mean = self.mean_output(hidden).view(b, s, -1, h, w)
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                self.output_dist.log_var = torch.clamp(self.trainable_log_var.view(1, 1, 1, 1, 1).repeat(b, s, 3, h, w), -7, 15)
            else:
                self.output_dist.log_var = torch.clamp(self.log_var_output(hidden).view(b, s, -1, h, w), -7, 15)
        return self.output_dist

    def kl_divergences(self, averaged=False):
        """Returns a list containing kl divergences at each level."""
        kl = []
        for latent_level in self.levels:
            kl.append(latent_level.latent.kl_divergence(analytical=False).sum(4).sum(3).sum(2).mean(1))
        if averaged:
            [level_kl.mean(dim=0) for level_kl in kl]
        else:
            return kl

    def conditional_log_likelihoods(self, input, averaged=False):
        """Returns the conditional likelihood."""
        if len(input.data.shape) == 4:
            input = input.unsqueeze(1) # add sample dimension
        # log_prob = self.output_dist.log_prob(sample=input) - np.log(256.)
        log_prob = self.output_dist.log_prob(sample=input).sub_(np.log(256.))
        log_prob = log_prob.sum(4).sum(3).sum(2).mean(1)
        if averaged:
            log_prob = log_prob.mean(dim=0)
        return log_prob

    def elbo(self, input, averaged=False):
        """Returns the ELBO."""
        cond_like = self.conditional_log_likelihoods(input)
        kl = sum(self.kl_divergences())
        lower_bound = cond_like - kl
        if averaged:
            return lower_bound.mean(dim=0)
        else:
            return lower_bound

    def losses(self, input, averaged=False):
        """Returns all losses."""
        cond_log_like = self.conditional_log_likelihoods(input)
        kl = self.kl_divergences()
        lower_bound = cond_log_like - sum(kl)
        if averaged:
            return lower_bound.mean(dim=0), cond_log_like.mean(dim=0), [level_kl.mean(dim=0) for level_kl in kl]
        else:
            return lower_bound, cond_log_like, kl

    def inference_model_parameters(self):
        """Returns a list containing all parameters in the encoder."""
        params = []
        for level in self.levels:
            params.extend(level.inference_model_parameters())
        return params

    def generative_model_parameters(self):
        """Returns a list containing all parameters in the decoder."""
        params = []
        for level in self.levels:
            params.extend(level.generative_model_parameters())
        params.extend(list(self.output_decoder.parameters()))
        params.extend(list(self.mean_output.parameters()))
        if self.output_distribution == 'gaussian':
            if self.constant_variances:
                params.append(self.trainable_log_var)
            else:
                params.extend(list(self.log_var_output.parameters()))
        return params

    def approx_posterior_parameters(self):
        """Returns the approximate posterior estimates."""
        params = []
        for latent_level in self.levels:
            params.extend(list(latent_level.latent.approx_posterior_parameters()))
        return params

    def step(self):
        """
        Steps the model forward in time by setting the previous approximate
        posterior to be the current appproximate posterior. Then sets priors
        and approximate posteriors for the new time step.
        """
        for latent_level in self.levels:
            latent_level.latent.step()
        self.generate(gen=True)
        self.reset_approx_posterior()

    def reset_approx_posterior(self):
        """Resets the approximate posterior estimates."""
        for latent_level in self.levels:
            latent_level.latent.reset_approx_posterior()

    def reset_prior(self):
        """Resets the prior estimates."""
        for latent_level in self.levels:
            latent_level.latent.reset_prior()

    def reinitialize_variables(self, output_dims):
        """
        Re-initialize the latent variables and ensure that they have the correct
        output size. Should always be done at the beginning of a sequence.
        Initializes the previous approximate posterior, prior, and approximate
        posterior.
        """
        for latent_level in self.levels:
            latent_level.latent.reinitialize_variable(output_dims)
        self.generate(gen=True)
        self.reset_approx_posterior()
