import torch
import torch.nn as nn
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
        self.inference_procedure = model_config['inference_procedure'].lower()
        latent_config = {}
        level_config = {}
        latent_config['inference_procedure'] = 'direct' # hard coded because we handle inference here in model
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
        else:
            raise Exception('VRNN model type must be one of 1) timit, 2) \
                            blizzard, or 3) iam_ondb. Invalid model \
                            type: ' + model_type + '.')

        # LSTM
        lstm_config = {'n_layers': 1, 'n_units': lstm_units, 'n_in': x_units + z_units}
        self.lstm = LSTMNetwork(lstm_config)

        # x model
        x_config = {'n_in': x_dim, 'n_units': x_units,
                    'n_layers': hidden_layers, 'non_linearity': 'relu'}
        self.x_model = FullyConnectedNetwork(x_config)

        # z model
        z_config = {'n_in': z_dim, 'n_units': z_units,
                    'n_layers': hidden_layers, 'non_linearity': 'relu'}
        self.z_model = FullyConnectedNetwork(z_config)

        # latent level (encoder model and prior model)
        inf_config = {'n_in': lstm_units + x_units, 'n_units': encoder_units,
                      'n_layers': hidden_layers, 'non_linearity': 'relu'}
        level_config['inference_config'] = inf_config
        gen_config = {'n_in': lstm_units, 'n_units': prior_units,
                      'n_layers': hidden_layers, 'non_linearity': 'relu'}
        level_config['generative_config'] = gen_config
        latent_config['n_variables'] = z_dim
        latent_config['n_in'] = (encoder_units, prior_units)
        level_config['latent_config'] = latent_config
        latent = FullyConnectedLatentLevel(level_config)
        self.latent_levels = nn.ModuleList([latent])

        # decoder
        decoder_config = {'n_in': lstm_units + z_units, 'n_units': decoder_units,
                          'n_layers': hidden_layers, 'non_linearity': 'relu'}
        self.decoder_model = FullyConnectedNetwork(decoder_config)

        self.output_dist = Normal()
        self.output_mean = FullyConnectedLayer({'n_in': decoder_units, 'n_out': x_dim})
        # self.output_log_var = FullyConnectedLayer({'n_in': decoder_units, 'n_out': x_dim})
        self.output_dist.log_var = nn.Parameter(torch.zeros(64, 1, 200))

    def _get_encoding_form(self, observation):
        """
        Gets the appropriate input form for the inference procedure.

        Args:
            observation (Variable, tensor): the input observation
        """
        if self.inference_procedure == 'direct':
            return observation
        else:
            raise NotImplementedError

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        self._x_enc = self.x_model(self._get_encoding_form(observation))
        self.latent_levels[0].infer(torch.cat([self._x_enc, self._prev_h], dim=1))

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        # TODO: handle sampling dimension reshape
        z = self.latent_levels[0].generate(self._prev_h.unsqueeze(1), gen=gen, n_samples=n_samples)
        self._z_enc = self.z_model(z)
        self._h = self.lstm(torch.cat([self._x_enc, self._z_enc[:, 0]], dim=1))
        output = self.decoder_model(torch.cat([self._z_enc, self._prev_h.unsqueeze(1)], dim=2))
        self.output_dist.mean = self.output_mean(output)
        # self.output_dist.log_var = self.output_log_var(output)
        return self.output_dist.sample()

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        # TODO: handle sampling dimension reshape better
        self._prev_h = self._h
        self._h = None

    def re_init(self, input):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        self.lstm.re_init(input)
        self._prev_h = self.lstm.layers[0].hidden_state
        self._h = None
        self.latent_levels[0].re_init()


    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.x_model.parameters()))
        params.extend(list(self.latent_levels[0].inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.lstm.parameters()))
        params.extend(list(self.latent_levels[0].generative_parameters()))
        params.extend(list(self.z_model.parameters()))
        params.extend(list(self.decoder_model.parameters()))
        params.extend(list(self.output_mean.parameters()))
        # params.extend(list(self.output_log_var.parameters()))
        params.append(self.output_dist.log_var)
        return params
