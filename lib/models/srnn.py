import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import FullyConnectedLatentLevel
from lib.modules.networks import LSTMNetwork, FullyConnectedNetwork
from lib.modules.layers import FullyConnectedLayer
from lib.distributions import Normal


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
        pass

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
        pass

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        pass

    def step(self, n_samples=1):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        pass

    def re_init(self, input):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        pass

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
