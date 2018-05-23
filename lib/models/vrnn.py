import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import FullyConnectedLatentLevel
from lib.modules.networks import LSTMNetwork, FullyConnectedNetwork
from lib.modules.layers import FullyConnectedLayer
from lib.distributions import Normal


class VRNN(LatentVariableModel):
    """
    Variational recurrent neural network (VRNN) from "A Recurrent Latent
    Variable Model for Sequential Data," Chung et al., 2015.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(VRNN, self).__init__(model_config)
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
        # hard coded because we handle inference here in the model
        level_config['inference_procedure'] = 'direct'

        if model_type == 'timit':
            lstm_units = 2000
            encoder_units = 500
            prior_units = 500
            decoder_units = 600
            x_units = 600
            z_units = 500
            hidden_layers = 4
            x_dim = 200
            z_dim = 200
            self.output_interval = 0.0018190742
        elif model_type == 'blizzard':
            lstm_units = 4000
            encoder_units = 500
            prior_units = 500
            decoder_units = 600
            x_units = 600
            z_units = 500
            hidden_layers = 4
            x_dim = 200
            z_dim = 200
            # TODO: check if this is correct
            self.output_interval = 0.0018190742
        elif model_type == 'iam_ondb':
            lstm_units = 1200
            encoder_units = 150
            prior_units = 150
            decoder_units = 250
            x_units = 250
            z_units = 150
            hidden_layers = 1
            x_dim = 3
            z_dim = 50
        elif model_type == 'bball':
            lstm_units = 1000
            encoder_units = 200
            prior_units = 200
            decoder_units = 200
            x_units = 200
            z_units = 200
            hidden_layers = 2
            x_dim = 2
            z_dim = 50
            self.output_interval = Variable(torch.from_numpy(np.array([1e-5 / 94., 1e-5 / 50.]).astype('float32')), requires_grad=False).cuda()
        else:
            raise Exception('VRNN model type must be one of 1) timit, 2) \
                            blizzard, 3) iam_ondb, or 4) bball. Invalid model \
                            type: ' + model_type + '.')

        # LSTM
        lstm_config = {'n_layers': 1, 'n_units': lstm_units, 'n_in': x_units + z_units}
        self.lstm = LSTMNetwork(lstm_config)

        # x model
        x_config = {'n_in': x_dim, 'n_units': x_units,
                    'n_layers': hidden_layers, 'non_linearity': 'relu'}
        self.x_model = FullyConnectedNetwork(x_config)

        # inf model
        if self.modified:
            if self.inference_procedure in ['direct', 'gradient', 'error']:
                # set the input encoding size
                if self.inference_procedure == 'direct':
                    input_dim = x_dim
                elif self.inference_procedure == 'gradient':
                    input_dim = 4 * z_dim
                    if model_config['concat_observation']:
                        input_dim += x_dim
                elif self.inference_procedure == 'error':
                    input_dim = x_dim + 3 * z_dim
                    if model_config['concat_observation']:
                        input_dim += x_dim
                else:
                    raise NotImplementedError

                encoder_units = 1024
                inf_config = {'n_in': input_dim, 'n_units': encoder_units,
                              'n_layers': 2, 'non_linearity': 'elu'}
                inf_config['connection_type'] = 'highway'
                # self.inf_model = FullyConnectedNetwork(inf_config)
            else:
                inf_config = None
                latent_config['inf_lr'] = model_config['learning_rate']
        else:
            inf_input_units = lstm_units + x_units
            inf_config = {'n_in': inf_input_units, 'n_units': encoder_units,
                          'n_layers': hidden_layers, 'non_linearity': 'relu'}

        # latent level (encoder model and prior model)
        level_config['inference_config'] = inf_config
        gen_config = {'n_in': lstm_units, 'n_units': prior_units,
                      'n_layers': hidden_layers, 'non_linearity': 'relu'}
        level_config['generative_config'] = gen_config
        latent_config['n_variables'] = z_dim
        latent_config['n_in'] = (encoder_units, prior_units)
        # latent_config['n_in'] = (encoder_units+input_dim, prior_units)
        level_config['latent_config'] = latent_config
        latent = FullyConnectedLatentLevel(level_config)
        self.latent_levels = nn.ModuleList([latent])

        # z model
        z_config = {'n_in': z_dim, 'n_units': z_units,
                    'n_layers': hidden_layers, 'non_linearity': 'relu'}
        self.z_model = FullyConnectedNetwork(z_config)

        # decoder
        decoder_config = {'n_in': lstm_units + z_units, 'n_units': decoder_units,
                          'n_layers': hidden_layers, 'non_linearity': 'relu'}
        self.decoder_model = FullyConnectedNetwork(decoder_config)

        self.output_dist = Normal()
        self.output_mean = FullyConnectedLayer({'n_in': decoder_units, 'n_out': x_dim})
        if model_config['global_output_log_var']:
            self.output_log_var = nn.Parameter(torch.zeros(x_dim))
        else:
            self.output_log_var = FullyConnectedLayer({'n_in': decoder_units, 'n_out': x_dim})

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
        self._x_enc = self.x_model(observation)
        if self.modified:
            enc = self._get_encoding_form(observation)
            self.latent_levels[0].infer(enc)
        else:
            inf_input = self._x_enc
            prev_h = self._prev_h
            # prev_h = prev_h.detach() if self._detach_h else prev_h
            enc = torch.cat([inf_input, prev_h], dim=1)
            self.latent_levels[0].infer(enc)

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        # TODO: handle sampling dimension
        # possibly detach the hidden state, preventing backprop
        prev_h = self._prev_h.unsqueeze(1)
        prev_h = prev_h.detach() if self._detach_h else prev_h

        # generate the prior
        z = self.latent_levels[0].generate(prev_h, gen=gen, n_samples=n_samples)

        # transform z through the z model
        b, s, _ = z.data.shape
        self._z_enc = self.z_model(z.view(b*s, -1)).view(b, s, -1)

        # pass encoded z and previous h through the decoder model
        dec = torch.cat([self._z_enc, prev_h.repeat(1, s, 1)], dim=2)
        b, s, _ = dec.data.shape
        output = self.decoder_model(dec.view(b*s, -1)).view(b, s, -1)

        # get the output mean and log variance
        self.output_dist.mean = self.output_mean(output)
        if self.model_config['global_output_log_var']:
            b, s = output.data.shape[0], output.data.shape[1]
            log_var = self.output_log_var.view(1, 1, -1).repeat(b, s, 1)
            self.output_dist.log_var = torch.clamp(log_var, min=-20)
        else:
            self.output_dist.log_var = torch.clamp(self.output_log_var(output), min=-20.)

        return self.output_dist.sample()

    def step(self, n_samples=1):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        # TODO: handle sampling dimension
        self._prev_h = self.lstm(torch.cat([self._x_enc, self._z_enc[:, 0]], dim=1))
        prev_h = self._prev_h.unsqueeze(1)
        self.lstm.step()
        # get the prior, use it to initialize the approximate posterior
        self.latent_levels[0].generate(prev_h, gen=True, n_samples=n_samples)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def re_init(self, input):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        # re-initialize the LSTM hidden and cell states
        self.lstm.re_init(input)
        # set the previous hidden state, add sample dimension
        self._prev_h = self.lstm.layers[0].hidden_state
        prev_h = self._prev_h.unsqueeze(1)
        # get the prior, use it to initialize the approximate posterior
        self.latent_levels[0].generate(prev_h, gen=True, n_samples=1)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = nn.ParameterList()
        if self.inference_procedure != 'sgd':
            params.extend(list(self.latent_levels[0].inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.lstm.parameters()))
        params.extend(list(self.latent_levels[0].generative_parameters()))
        params.extend(list(self.x_model.parameters()))
        params.extend(list(self.z_model.parameters()))
        params.extend(list(self.decoder_model.parameters()))
        params.extend(list(self.output_mean.parameters()))
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
