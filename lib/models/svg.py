import torch
import torch.nn as nn
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import LSTMLatentLevel
from lib.modules.networks import LSTMNetwork, FullyConnectedNetwork
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
        latent_config['normalize_samples'] = model_config['normalize_latent_samples']
        latent_config['inference_procedure'] = self.inference_procedure
         # hard coded because we handle inference here in the model
        level_config['inference_procedure'] = 'direct'

        if not self.modified:
            level_config['inference_config'] = {'n_layers': 1, 'n_units': 256, 'n_in': 128}
            latent_config['n_in'] = (256, 256) # number of encoder, decoder units
        else:
            level_config['inference_config'] = None
            latent_config['n_in'] = [None, 256] # number of encoder, decoder units
        level_config['generative_config'] = None

        if model_type == 'sm_mnist':
            from lib.modules.networks.dcgan_64 import encoder, decoder
            self.n_input_channels = 1
            self.encoder = encoder(128, self.n_input_channels)
            self.decoder = decoder(128, self.n_input_channels)
            self.output_dist = Bernoulli()
            latent_config['n_variables'] = 10
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
            self.n_input_channels = 1
            self.encoder = encoder(128, self.n_input_channels)
            if model_config['global_output_log_var']:
                output_channels = self.n_input_channels
                self.output_log_var = nn.Parameter(torch.zeros(self.n_input_channels, 64, 64))
            else:
                output_channels = 2 * self.n_input_channels
            self.decoder = decoder(128, output_channels)
            self.output_dist = Normal()
            latent_config['n_variables'] = 512
            if self.modified:
                if self.inference_procedure == 'direct':
                    # another convolutional encoder
                    self.inf_encoder = encoder(128, self.n_input_channels)
                    # fully-connected inference model
                    inf_config = {'n_layers': 2,
                                  'n_units': 256,
                                  'n_in': 128,
                                  'non_linearity': 'relu'}
                    self.inf_model = FullyConnectedNetwork(inf_config)
                    latent_config['n_in'][0] = 256
                elif self.inference_procedure == 'gradient':
                    # fully-connected encoder / latent inference model
                    n_units = 1024
                    inf_config = {'n_layers': 1,
                                  'n_units': n_units,
                                  'n_in': 4 * latent_config['n_variables'],
                                  'non_linearity': 'elu',
                                  'connection_type': 'highway'}
                    if model_config['concat_observation']:
                        inf_config['n_in'] += (self.n_input_channels * 64 * 64)
                    self.inf_model = FullyConnectedNetwork(inf_config)
                    latent_config['n_in'][0] = n_units
                    latent_config['update_type'] = model_config['update_type']
                elif self.inference_procedure == 'error':
                    # convolutional observation error encoder
                    obs_error_enc_config = {'n_layers': 3,
                                            'n_filters': 64,
                                            'n_in': self.n_input_channels,
                                            'filter_size': 3,
                                            'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        obs_error_enc_config['n_in'] += self.n_input_channels
                    self.obs_error_enc = ConvolutionalNetwork(obs_error_enc_config)
                    # fully-connected error encoder (latent error + params + encoded observation errors)
                    inf_config = {'n_layers': 3,
                                  'n_units': 1024,
                                  'n_in': 4 * latent_config['n_variables'],
                                  'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        inf_config['n_in'] += (self.n_input_channels * 64 * 64)
                    self.inf_model = FullyConnectedNetwork(inf_config)
                    latent_config['n_in'][0] = 1024
                    latent_config['update_type'] = model_config['update_type']
                else:
                    raise NotImplementedError

        elif model_type == 'bair_robot_pushing':
            from lib.modules.networks.vgg_64 import encoder, decoder
            self.n_input_channels = 3
            self.encoder = encoder(128, self.n_input_channels)
            if model_config['global_output_log_var']:
                output_channels = self.n_input_channels
                self.output_log_var = nn.Parameter(torch.zeros(self.n_input_channels, 64, 64))
            else:
                output_channels = 2 * self.n_input_channels
            self.decoder = decoder(128, output_channels)
            self.output_dist = Normal()
            latent_config['n_variables'] = 64
            if self.modified:
                if self.inference_procedure == 'direct':
                    # another convolutional encoder
                    self.inf_encoder = encoder(128, self.n_input_channels)
                    # fully-connected inference model
                    inf_config = {'n_layers': 2,
                                  'n_units': 256,
                                  'n_in': 128,
                                  'non_linearity': 'relu'}
                    self.inf_model = FullyConnectedNetwork(inf_config)
                    latent_config['n_in'][0] = 256
                elif self.inference_procedure == 'gradient':
                    # fully-connected encoder / latent inference model
                    inf_config = {'n_layers': 3,
                                  'n_units': 1024,
                                  'n_in': 4 * latent_config['n_variables'],
                                  'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        inf_config['n_in'] += (self.n_input_channels * 64 * 64)
                    self.inf_model = FullyConnectedNetwork(inf_config)
                    latent_config['n_in'][0] = 1024
                    latent_config['update_type'] = model_config['update_type']
                elif self.inference_procedure == 'error':
                    # convolutional observation error encoder
                    obs_error_enc_config = {'n_layers': 3,
                                            'n_filters': 64,
                                            'n_in': self.n_input_channels,
                                            'filter_size': 3,
                                            'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        obs_error_enc_config['n_in'] += self.n_input_channels
                    self.obs_error_enc = ConvolutionalNetwork(obs_error_enc_config)
                    # fully-connected error encoder (latent error + params + encoded observation errors)
                    inf_config = {'n_layers': 3,
                                  'n_units': 1024,
                                  'n_in': 4 * latent_config['n_variables'],
                                  'non_linearity': 'relu'}
                    if model_config['concat_observation']:
                        inf_config['n_in'] += (self.n_input_channels * 64 * 64)
                    self.inf_model = FullyConnectedNetwork(inf_config)
                    latent_config['n_in'][0] = 1024
                    latent_config['update_type'] = model_config['update_type']
                else:
                    raise NotImplementedError
        else:
            raise Exception('SVG model type must be one of 1) sm_mnist, 2) \
                            kth_action, or 3) bair_robot_pushing. Invalid model \
                            type: ' + model_type + '.')

        # construct a recurrent latent level
        level_config['latent_config'] = latent_config
        self.latent_levels = nn.ModuleList([LSTMLatentLevel(level_config)])

        self.prior_lstm = LSTMNetwork({'n_layers': 1, 'n_units': 256, 'n_in': 128})

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
            # TODO: figure out proper normalization for observation error
            errors = [self._output_error(observation), self.latent_levels[0].latent.error()]
            # normalize
            for ind, error in enumerate(errors):
                mean = error.mean(dim=0, keepdim=True)
                std = error.std(dim=0, keepdim=True)
                errors[ind] = (error - mean) / (std + 1e-5)
            # concatenate
            params = torch.cat(self.latent_levels[0].latent.approx_posterior_parameters(), dim=1)
            latent_error_params = torch.cat([errors[1], params], dim=1)
            if self.model_config['concat_observation']:
                latent_error_params = torch.cat([latent_error_params, observation], dim=1)
            return errors[0], latent_error_params
        else:
            raise NotImplementedError

    def _output_error(self, observation, averaged=True):
        """
        Calculates Gaussian error for encoding.

        Args:
            observation (tensor): observation to use for error calculation
        """
        # get the output mean and log variance
        output_mean = self.output_dist.mean.detach()
        output_log_var = self.output_dist.log_var.detach()
        # repeat the observation along the sample dimension
        n_samples = output_mean.data.shape[1]
        observation = observation.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
        # calculate the precision-weighted observation error
        n_error = (observation - output_mean) / (output_log_var.exp() + 1e-7)
        if averaged:
            # average along the sample dimension
            n_error = n_error.mean(dim=1)
        return n_error

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        if self.modified:
            if not self._obs_encoded:
                # encode the observation (to be used at the next time step)
                self._h, self._skip = self.encoder(observation - 0.5)
                self._obs_encoded = True

            enc = self._get_encoding_form(observation)

            if self.inference_procedure == 'direct':
                # separate encoder model
                enc_h, _ = self.inf_encoder(enc)
                enc_h = self.inf_model(enc_h)

            elif self.inference_procedure == 'gradient':
                # encode through the inference model
                enc_h = self.inf_model(enc)

            elif self.inference_procedure == 'error':
                # encode the error and flatten it
                enc_error = self.obs_error_enc(enc[0])
                enc_error = enc_error.view(enc_error.data.shape[0], -1)
                # concatenate the error with the rest of the terms
                enc = torch.cat([enc_error, enc[1]], dim=1)
                # encode through the inference model
                enc_h = self.inf_model(enc)

            self.latent_levels[0].infer(enc_h)

        else:
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
        batch_size = self._prev_h.data.shape[0]

        # get the previous h and skip
        prev_h = self._prev_h.unsqueeze(1)
        prev_skip = [0. * _prev_skip.repeat(n_samples, 1, 1, 1) for _prev_skip in self._prev_skip]

        # detach prev_h and prev_skip if necessary
        if self._detach_h:
            prev_h = prev_h.detach()
            prev_skip = [_prev_skip.detach() for _prev_skip in prev_skip]

        # get the prior input, detach if necessary
        gen_input = self._gen_input
        gen_input = gen_input.detach() if self._detach_h else gen_input

        # sample the latent variables
        z = self.latent_levels[0].generate(gen_input.unsqueeze(1), gen=gen, n_samples=n_samples)

        # pass through the decoder
        decoder_input = torch.cat([z, prev_h], dim=2).view(batch_size * n_samples, -1)
        g = self.decoder_lstm(decoder_input, detach=self._detach_h)
        g = self.decoder_lstm_output(g)
        output = self.decoder([g, prev_skip])
        b, _, h, w = output.data.shape

        # get the output mean and log variance
        if self.model_config['global_output_log_var']:
            # repeat along batch and sample dimensions
            output = output.view(b, -1, self.n_input_channels, h, w)
            log_var = self.output_log_var.unsqueeze(0).unsqueeze(0).repeat(batch_size, n_samples, 1, 1, 1)
            self.output_dist.log_var = torch.clamp(log_var, min=-10)
        else:
            output = output.view(b, -1, 2 * self.n_input_channels, h, w)
            self.output_dist.log_var = torch.clamp(output[:, :, self.n_input_channels:, :, :], min=-10)

        self.output_dist.mean = output[:, :, :self.n_input_channels, :, :].sigmoid()

        return torch.clamp(self.output_dist.sample(), 0., 1.)

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        # TODO: set n_samples in a smart way
        # step the lstms and latent level
        self.latent_levels[0].step()
        self.prior_lstm.step()
        self.decoder_lstm.step()
        # copy over the hidden and skip variables
        self._prev_h = self._h
        self._prev_skip = self._skip
        # clear the current hidden and skip variables, set the flag
        self._h = self._skip = None
        self._obs_encoded = False
        # use the prior lstm to get generative model input
        self._gen_input = self.prior_lstm(self._prev_h.unsqueeze(1), detach=False)
        # set the prior and approximate posterior
        self.latent_levels[0].generate(self._gen_input.detach().unsqueeze(1), gen=True, n_samples=1)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def re_init(self, input):
        """
        Method for reinitializing the state (distributions and hidden states).

        Args:
            input (Variable, Tensor): contains observation at t = -1
        """
        # TODO: set n_samples in a smart way
        # flag to encode the hidden state for later decoding
        self._obs_encoded = False
        # re-initialize the lstms and distributions
        self.latent_levels[0].re_init()
        self.prior_lstm.re_init(input)
        self.decoder_lstm.re_init(input)
        # clear the hidden state and skip
        self._h = self._skip = None
        # encode this input to set the previous h and skip
        self._prev_h, self._prev_skip = self.encoder(input - 0.5)
        # set the prior and approximate posterior
        self._gen_input = self.prior_lstm(self._prev_h, detach=False)
        self.latent_levels[0].generate(self._gen_input.unsqueeze(1), gen=True, n_samples=1)
        self.latent_levels[0].latent.re_init_approx_posterior()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = nn.ParameterList()
        if self.modified:
            params.extend(list(self.inf_model.parameters()))
            if self.inference_procedure == 'direct':
                params.extend(list(self.inf_encoder.parameters()))
            elif self.inference_procedure == 'gradient':
                pass # no other inference parameters
            elif self.inference_procedure == 'error':
                params.extend(list(self.obs_error_enc.parameters()))
            else:
                raise NotImplementedError
        else:
            params.extend(list(self.encoder.parameters()))
        params.extend(list(self.latent_levels[0].inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = nn.ParameterList()
        if self.modified:
            params.extend(list(self.encoder.parameters()))
        params.extend(list(self.prior_lstm.parameters()))
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.latent_levels[0].generative_parameters()))
        params.extend(list(self.decoder_lstm.parameters()))
        params.extend(list(self.decoder_lstm_output.parameters()))
        if self.model_config['global_output_log_var']:
            params.append(self.output_log_var)
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
