import torch
import torch.nn as nn
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import LSTMLatentLevel
from lib.modules.networks import LSTMNetwork
from lib.modules.layers import FullyConnectedLayer
from lib.distributions import Normal, Bernoulli


class SVG(LatentVariableModel):
    """
    Stochastic video generation (SVG) model from "Stochastic Video Generation
    with a Learned Prior," Denton & Fergus, 2018.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(SVG, self).__init__(model_config)
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Method for constructing SVG model using the model configuration file.

        Args:
            model_config (dict): dictionary containing model configuration params
        """
        model_type = model_config['model_type'].lower()
        self.modified = model_config['modified']
        self.inference_procedure = model_config['inference_procedure'].lower()

        level_config = {}
        latent_config = {}

        latent_config['n_in'] = (256, 256) # number of encoder, decoder units
        latent_config['inference_procedure'] = self.inference_procedure
         # hard coded because we handle inference here in the model
        level_config['inference_procedure'] = 'direct'

        if not self.modified:
            level_config['inference_config'] = {'n_layers': 1, 'n_units': 256, 'n_in': 128}
        level_config['generative_config'] = {'n_layers': 1, 'n_units': 256, 'n_in': 128}

        if model_type == 'sm_mnist':
            from lib.modules.networks.dcgan_64 import encoder, decoder
            self.encoder = encoder(128, 1)
            self.decoder = decoder(128, 1)
            self.output_dist = Bernoulli()
            latent_config['n_variables'] = 10
            level_config['latent_config'] = latent_config
            if self.modified:
                if self.inference_procedure == 'direct':
                    pass
                elif self.inference_procedure == 'gradient':
                    pass
                elif self.inference_procedure == 'error':
                    pass
                else:
                    raise NotImplementedError

        elif model_type == 'kth_actions':
            from lib.modules.networks.vgg_64 import encoder, decoder
            self.encoder = encoder(128, 1)
            if model_config['global_output_log_var']:
                output_channels = 1
                self.output_log_var = nn.Parameter(torch.zeros(1, 64, 64))
            else:
                output_channels = 2
            self.decoder = decoder(128, output_channels)
            self.output_dist = Normal()
            latent_config['n_variables'] = 32
            level_config['latent_config'] = latent_config
            if self.modified:
                if self.inference_procedure == 'direct':
                    pass
                elif self.inference_procedure == 'gradient':
                    pass
                elif self.inference_procedure == 'error':
                    pass
                else:
                    raise NotImplementedError

        elif model_type == 'bair_robot_pushing':
            from lib.modules.networks.vgg_64 import encoder, decoder
            self.encoder = encoder(128, 3)
            if model_config['global_output_log_var']:
                output_channels = 3
                self.output_log_var = nn.Parameter(torch.zeros(3, 64, 64))
            else:
                output_channels = 6
            self.decoder = decoder(128, output_channels)
            self.output_dist = Normal()
            latent_config['n_variables'] = 64
            level_config['latent_config'] = latent_config
            if self.modified:
                if self.inference_procedure == 'direct':
                    # another convolutional encoder
                    self.inf_encoder = encoder(128, 3)
                    # fully-connected latent inference model
                    level_config['inference_config'] = {'n_layers': 2,
                                                        'n_units': 256,
                                                        'n_in': 128,
                                                        'non_linearity': 'relu'}
                elif self.inference_procedure == 'gradient':
                    # fully-connected encoder / latent inference model
                    level_config['inference_config'] = {'n_layers': 3,
                                                        'n_units': 1024,
                                                        'n_in': 4 * latent_config['n_variables'],
                                                        'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        level_config['inference_config']['n_in'] += (3 * 64 * 64)
                elif self.inference_procedure == 'error':
                    # convolutional observation error encoder
                    obs_error_enc_config = {'n_layers': 3,
                                            'n_filters': 64,
                                            'n_in': 3,
                                            'filter_size': 3,
                                            'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        obs_error_enc_config['n_in'] += 3
                    self.obs_error_enc = ConvolutionalNetwork(obs_error_enc_config)
                    # fully-connected error encoder (latent + params + encoded observation errors)
                    level_config['inference_config'] = {'n_layers': 3,
                                                        'n_units': 1024,
                                                        'n_in': 4 * latent_config['n_variables'],
                                                        'non_linearity': 'relu'}
                else:
                    raise NotImplementedError
        else:
            raise Exception('SVG model type must be one of 1) sm_mnist, 2) \
                            kth_action, or 3) bair_robot_pushing. Invalid model \
                            type: ' + model_type + '.')

        # construct the latent level
        if self.modified:
            # use a fully-connected latent level
            self.latent_error_enc = FullyConnectedNetwork(latent_error_enc_config)
        else:
            # use a recurrent latent level
            self.latent_levels = nn.ModuleList([LSTMLatentLevel(level_config)])

        self.decoder_lstm = LSTMNetwork({'n_layers': 2, 'n_units': 256,
                                         'n_in': 128 + latent_config['n_variables']})
        self.decoder_lstm_output = FullyConnectedLayer({'n_in': 256, 'n_out': 128,
                                                        'non_linearity': 'tanh'})


        self.output_interval = 1./256

    def _get_encoding_form(self, observation):
        """
        Gets the appropriate input form for the inference procedure.

        Args:
            observation (Variable, tensor): the input observation
        """
        if self.inference_procedure == 'direct':
            return observation - 0.5
        if self.inference_procedure == 'gradient':
            raise NotImplementedError
        elif self.inference_procedure == 'error':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        observation = self._get_encoding_form(observation)
        self._h, self._skip = self.encoder(observation)
        self.latent_levels[0].infer(self._h)

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        # generate the prior, sample from the latent variables
        batch_size = self._prev_h.data.shape[0]
        prev_h = self._prev_h.unsqueeze(1)
        prev_skip = [_prev_skip.repeat(n_samples, 1, 1, 1) for _prev_skip in self._prev_skip]

        z = self.latent_levels[0].generate(None, gen=gen, n_samples=n_samples)
        g = self.decoder_lstm(torch.cat([z, prev_h], dim=2).view(batch_size * n_samples, -1))
        g = self.decoder_lstm_output(g)
        output = self.decoder([g, prev_skip])
        b, _, h, w = output.data.shape
        output = output.view(b, -1, 6, h, w)
        self.output_dist.mean = torch.nn.Sigmoid()(output[:, :, :3, :, :])
        self.output_dist.log_var = output[:, :, 3:, :, :]
        return torch.clamp(self.output_dist.sample(), 0., 1.)

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        # TODO: set n_samples in a smart way
        self.latent_levels[0].step()
        self.decoder_lstm.step()
        self._prev_h = self._h
        self._prev_skip = self._skip
        self._h = self._skip = None
        self.latent_levels[0].generate(self._prev_h.unsqueeze(1), gen=True, n_samples=1)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def re_init(self, input):
        """
        Method for reinitializing the state (distributions and hidden states).

        Args:
            input (Variable, Tensor): contains observation at t = -1
        """
        # TODO: set n_samples in a smart way
        # re-initialize hidden states and distributions
        self.latent_levels[0].re_init()
        self.decoder_lstm.re_init()
        # clear the hidden state
        self._h = self._skip = None
        # set the prior and approx. posterior
        self._prev_h, self._prev_skip = self.encoder(self._get_encoding_form(input))
        self.latent_levels[0].generate(self._prev_h.unsqueeze(1), gen=True, n_samples=1)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.encoder.parameters()))
        params.extend(list(self.latent_levels[0].inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.latent_levels[0].generative_parameters()))
        params.extend(list(self.decoder_lstm.parameters()))
        params.append(self.output_dist.log_var)
        return params

    def inference_mode(self):
        """
        Method to set the model's current mode.
        """
        self.mode = 'inference'
        if self.inference_procedure == 'direct':
            pass
        else:
            pass

    def generative_mode(self):
        """
        Method to set the model's current mode.
        """
        self.mode = 'generation'
        if self.inference_procedure == 'direct':
            pass
        else:
            pass
