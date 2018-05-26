import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import FullyConnectedLatentLevel
from lib.modules.networks import LSTMNetwork, FullyConnectedNetwork
from lib.modules.layers import FullyConnectedLayer
from lib.distributions import Normal, Bernoulli


class SRNN(LatentVariableModel):
    """
    Stochastic recurrent neural network (SRNN) from Fraccaro et al., 2016.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(SRNN, self).__init__(model_config)
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Args:
            model_config (dict): dictionary containing model configuration params
        """
        model_type = model_config['model_type'].lower()
        self.modified = model_config['modified']
        self.inference_procedure = model_config['inference_procedure'].lower()
        if not self.modified:
            assert self.inference_procedure == 'direct', 'The original model only supports direct inference.'
        self._detach_h = False
        latent_config = {}
        level_config = {}
        latent_config['inference_procedure'] = self.inference_procedure
        latent_config['normalize_samples'] = model_config['normalize_latent_samples']
        # hard coded because we handle inference here in the model
        level_config['inference_procedure'] = 'direct'

        if model_type == 'timit':
            lstm_units = 1024
            x_dim = 200
            z_dim = 256
            n_layers = 2
            n_units = 512
            # Gaussian output
            self.output_interval = 0.0018190742
            self.output_dist = Normal()
            self.output_mean = FullyConnectedLayer({'n_in': n_units,
                                                    'n_out': x_dim})
            if model_config['global_output_log_var']:
                self.output_log_var = nn.Parameter(torch.zeros(x_dim))
            else:
                self.output_log_var = FullyConnectedLayer({'n_in': n_units,
                                                           'n_out': x_dim})
        elif model_type == 'midi':
            lstm_units = 300
            x_dim = 88
            z_dim = 100
            n_layers = 1
            n_units = 500
            # Bernoulli output
            self.output_dist = Bernoulli()
            self.output_mean = FullyConnectedLayer({'n_in': n_units,
                                                    'n_out': x_dim,
                                                    'non_linearity': 'sigmoid'})
            self.output_log_var = None
        else:
            raise Exception('SRNN model type must be one of 1) timit, 2) \
                             or 4) midi. Invalid model type: ' + model_type + '.')

        # LSTM
        lstm_config = {'n_layers': 1, 'n_units': lstm_units, 'n_in': x_dim}
        self.lstm = LSTMNetwork(lstm_config)

        # non_linearity = 'sigmoid'

        # latent level
        gen_config = {'n_in': lstm_units + z_dim, 'n_units': n_units,
                      'n_layers': n_layers, 'non_linearity': 'clipped_leaky_relu'}
        # gen_config = {'n_in': lstm_units + z_dim, 'n_units': n_units,
        #               'n_layers': n_layers, 'non_linearity': non_linearity}
        level_config['generative_config'] = gen_config
        level_config['inference_config'] = None
        latent_config['n_variables'] = z_dim

        if self.modified:
            inf_model_units = 1024
            inf_model_layers = 2
            inf_model_config = {'n_in': 4 * z_dim, 'n_units': inf_model_units,
                                'n_layers': inf_model_layers, 'non_linearity': 'elu'}
            # inf_model_config = {'n_in': 4 * z_dim, 'n_units': inf_model_units,
            #                     'n_layers': inf_model_layers, 'non_linearity': non_linearity}
            if model_config['concat_observation']:
                inf_model_config['n_in'] += x_dim
            inf_model_config['connection_type'] = 'highway'
            latent_config['update_type'] = model_config['update_type']
        else:
            inf_model_units = n_units
            inf_model_config = {'n_in': lstm_units + x_dim, 'n_units': n_units,
                                'n_layers': n_layers, 'non_linearity': 'clipped_leaky_relu'}
            # inf_model_config = {'n_in': lstm_units + x_dim, 'n_units': n_units,
            #                     'n_layers': n_layers, 'non_linearity': non_linearity}
        self.inference_model = FullyConnectedNetwork(inf_model_config)
        latent_config['n_in'] = (inf_model_units, n_units)
        level_config['latent_config'] = latent_config
        latent = FullyConnectedLatentLevel(level_config)
        self.latent_levels = nn.ModuleList([latent])

        self._initial_z = nn.Parameter(torch.zeros(z_dim))

        # decoder
        decoder_config = {'n_in': lstm_units + z_dim, 'n_units': n_units,
                          'n_layers': 2, 'non_linearity': 'clipped_leaky_relu'}
        # decoder_config = {'n_in': lstm_units + z_dim, 'n_units': n_units,
        #                   'n_layers': 2, 'non_linearity': non_linearity}
        self.decoder_model = FullyConnectedNetwork(decoder_config)


    def _get_encoding_form(self, observation):
        """
        Gets the appropriate input form for the inference procedure.

        Args:
            observation (Variable, tensor): the input observation
        """
        if self.inference_procedure == 'direct':
            return observation

        elif self.inference_procedure == 'gradient':
            grads = self.latent_levels[0].latent.approx_posterior_gradients()

            # normalization
            if self.model_config['input_normalization'] in ['layer', 'batch']:
                norm_dim = 0 if self.model_config['input_normalization'] == 'batch' else 1
                for ind, grad in enumerate(grads):
                    mean = grad.mean(dim=norm_dim, keepdim=True)
                    std = grad.std(dim=norm_dim, keepdim=True)
                    grads[ind] = (grad - mean) / (std + 1e-7)
            grads = torch.cat(grads, dim=1)

            # concatenate with the parameters
            params = self.latent_levels[0].latent.approx_posterior_parameters()
            if self.model_config['norm_parameters']:
                if self.model_config['input_normalization'] in ['layer', 'batch']:
                    norm_dim = 0 if self.model_config['input_normalization'] == 'batch' else 1
                    for ind, param in enumerate(params):
                        mean = param.mean(dim=norm_dim, keepdim=True)
                        std = param.std(dim=norm_dim, keepdim=True)
                        params[ind] = (param - mean) / (std + 1e-7)
            params = torch.cat(params, dim=1)

            grads_params = torch.cat([grads, params], dim=1)

            # concatenate with the observation
            if self.model_config['concat_observation']:
                grads_params = torch.cat([grads_params, observation], dim=1)

            return grads_params

        elif self.inference_procedure == 'error':
            errors = [self._output_error(observation), self.latent_levels[0].latent.error()]

            # normalization
            if self.model_config['input_normalization'] in ['layer', 'batch']:
                norm_dim = 0 if self.model_config['input_normalization'] == 'batch' else 1
                for ind, error in enumerate(errors):
                    mean = error.mean(dim=0, keepdim=True)
                    std = error.std(dim=0, keepdim=True)
                    errors[ind] = (error - mean) / (std + 1e-7)
            errors = torch.cat(errors, dim=1)

            # concatenate with the parameters
            params = self.latent_levels[0].latent.approx_posterior_parameters()
            if self.model_config['norm_parameters']:
                if self.model_config['input_normalization'] in ['layer', 'batch']:
                    norm_dim = 0 if self.model_config['input_normalization'] == 'batch' else 1
                    for ind, param in enumerate(params):
                        mean = param.mean(dim=norm_dim, keepdim=True)
                        std = param.std(dim=norm_dim, keepdim=True)
                        params[ind] = (param - mean) / (std + 1e-7)
            params = torch.cat(params, dim=1)

            error_params = torch.cat([errors, params], dim=1)

            if self.model_config['concat_observation']:
                error_params = torch.cat([error_params, observation], dim=1)

            return error_params

        elif self.inference_procedure == 'sgd':
            grads = self.latent_levels[0].latent.approx_posterior_gradients()
            return grads

        else:
            raise NotImplementedError

    def _output_error(self, observation, averaged=True):
        """
        Calculates Gaussian error for encoding.

        Args:
            observation (tensor): observation to use for error calculation
        """
        output_mean = self.output_dist.mean.detach()
        output_log_var = self.output_dist.log_var.detach()
        n_samples = output_mean.data.shape[1]
        if len(observation.data.shape) == 2:
            observation = observation.unsqueeze(1).repeat(1, n_samples, 1)
        n_error = (observation - output_mean) / torch.exp(output_log_var + 1e-7)
        if averaged:
            n_error = n_error.mean(dim=1)
        return n_error

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        if self._x is None:
            # store the observation
            self._x = observation
        enc = self._get_encoding_form(observation)
        if self.modified:
            pass
        else:
            h = self._h
            enc = torch.cat([enc, h], dim=1)
        enc = self.inference_model(enc)
        self.latent_levels[0].infer(enc)

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        h = self._h.detach() if self._detach_h else self._h
        h = h.unsqueeze(1)
        prev_z = self._prev_z.detach() if self._detach_h else self._prev_z
        if prev_z.data.shape[1] != n_samples:
            prev_z = prev_z.repeat(1, n_samples, 1)

        gen_input = torch.cat([h.repeat(1, n_samples, 1), prev_z], dim=2)
        self._z = self.latent_levels[0].generate(gen_input, gen=gen, n_samples=n_samples)

        dec_input = torch.cat([h.repeat(1, n_samples, 1), self._z], dim=2)
        b, s, _ = dec_input.data.shape
        dec = self.decoder_model(dec_input.view(b * s, -1)).view(b, s, -1)

        output_mean = self.output_mean(dec)

        if self.output_log_var:
            # Gaussian output
            if self.model_config['global_output_log_var']:
                b, s = dec.data.shape[0], dec.data.shape[1]
                log_var = self.output_log_var.view(1, 1, -1).repeat(b, s, 1)
                self.output_dist.log_var = torch.clamp(log_var, min=-10)
            else:
                output_log_var = torch.clamp(self.output_log_var(dec), min=-10)
            self.output_dist = Normal(output_mean, output_log_var)
        else:
            # Bernoulli output
            self.output_dist = Bernoulli(output_mean)

        return self.output_dist.sample()

    def step(self, n_samples=1):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        # set the previous z
        self._prev_z = self._z
        s = self._prev_z.data.shape[1]

        # step the LSTM (using the previous observation)
        self._h = self.lstm(self._x)
        self.lstm.step()
        self._x = None

        # get the prior, use it to initialize the approximate posterior
        gen_input = torch.cat([self._h.unsqueeze(1).repeat(1, s, 1), self._prev_z], dim=2)
        self.latent_levels[0].generate(gen_input, gen=True, n_samples=n_samples)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def re_init(self, input):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        # re-initialize the LSTM hidden and cell states
        self.lstm.re_init(input)

        # set the previous hidden state, add sample dimension
        self._h = self.lstm(input)
        self.lstm.step()
        self._x = None

        self._z = None
        self._prev_z = self._initial_z.view(1, 1, -1).repeat(input.data.shape[0], 1, 1)

        # get the prior, use it to initialize the approximate posterior
        gen_input = torch.cat([self._h.unsqueeze(1), self._prev_z], dim=2)
        self.latent_levels[0].generate(gen_input, gen=True, n_samples=1)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = nn.ParameterList()
        if self.inference_procedure != 'sgd':
            params.extend(list(self.inference_model.parameters()))
            params.extend(list(self.latent_levels[0].inference_parameters()))
            params.append(self._initial_z)
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.lstm.parameters()))
        params.extend(list(self.latent_levels[0].generative_parameters()))
        params.extend(list(self.decoder_model.parameters()))
        params.extend(list(self.output_mean.parameters()))
        if self.output_log_var is not None:
            if self.model_config['global_output_log_var']:
                params.append(self.output_log_var)
            else:
                params.extend(list(self.output_log_var.parameters()))
        return params

    def inference_mode(self):
        """
        Method to set the model's current mode to inference.
        """
        self.latent_levels[0].latent.detach = False
        self._detach_h = True

    def generative_mode(self):
        """
        Method to set the model's current mode to generation.
        """
        self.latent_levels[0].latent.detach = True
        self._detach_h = False
