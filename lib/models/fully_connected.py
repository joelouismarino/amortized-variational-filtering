import torch
import torch.nn as nn
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import FullyConnectedLatentLevel
from lib.modules.networks import FullyConnectedNetwork
from lib.modules.layers import FullyConnectedLayer
from lib.distributions import Normal


class FullyConnectedDLVM(LatentVariableModel):
    """
    A generic fully connected dynamical latent variable model.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(FullyConnectedDLVM, self).__init__(model_config)
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Method for constructing SVG model using the model configuration file.

        Args:
            model_config (dict): dictionary containing model configuration params
        """
        pass

    # def __construct(self, arch):
    #     """Construct the model from the architecture dictionary."""
    #
    #     self.encoding_form = arch['encoding_form']
    #     self.constant_variances = arch['constant_prior_variances']
    #     self.batch_size = train_config['batch_size']
    #     self.kl_min = train_config['kl_min']
    #     self.concat_variables = arch['concat_variables']
    #     self.top_size = arch['top_size']
    #     self.input_size = np.prod(tuple(next(iter(data_loader))[0].size()[1:])).astype(int)
    #     assert train_config['output_distribution'] in ['bernoulli', 'gaussian'], 'Output distribution not recognized.'
    #     self.output_distribution = train_config['output_distribution']
    #     self.reconstruction = None
    #
    #     # construct the model
    #     self.levels = [None for _ in range(len(arch['n_latent']))]
    #     self.output_decoder = self.output_dist = self.mean_output = self.log_var_output = self.trainable_log_var = None
    #
    #     # these are the same across all latent levels
    #     encoding_form = arch['encoding_form']
    #     variable_update_form = arch['variable_update_form']
    #     const_prior_var = arch['constant_prior_variances']
    #
    #     encoder_arch = {}
    #     encoder_arch['non_linearity'] = arch['non_linearity_enc']
    #     encoder_arch['connection_type'] = arch['connection_type_enc']
    #     encoder_arch['batch_norm'] = arch['batch_norm_enc']
    #     encoder_arch['weight_norm'] = arch['weight_norm_enc']
    #     encoder_arch['dropout'] = arch['dropout_enc']
    #
    #     decoder_arch = {}
    #     decoder_arch['non_linearity'] = arch['non_linearity_dec']
    #     decoder_arch['connection_type'] = arch['connection_type_dec']
    #     decoder_arch['batch_norm'] = arch['batch_norm_dec']
    #     decoder_arch['weight_norm'] = arch['weight_norm_dec']
    #     decoder_arch['dropout'] = arch['dropout_dec']
    #
    #     # construct a DenseLatentLevel for each level of latent variables
    #     for level in range(len(arch['n_latent'])):
    #
    #         # get specifications for this level's encoder and decoder
    #         encoder_arch['n_in'] = self.encoder_input_size(level, arch)
    #         encoder_arch['n_units'] = arch['n_units_enc'][level]
    #         encoder_arch['n_layers'] = arch['n_layers_enc'][level]
    #
    #         decoder_arch['n_in'] = self.decoder_input_size(level, arch)
    #         decoder_arch['n_units'] = arch['n_units_dec'][level+1]
    #         decoder_arch['n_layers'] = arch['n_layers_dec'][level+1]
    #
    #         n_latent = arch['n_latent'][level]
    #         n_det = [arch['n_det_enc'][level], arch['n_det_dec'][level]]
    #
    #         learn_prior = True if arch['learn_top_prior'] else (level != len(arch['n_latent'])-1)
    #
    #         self.levels[level] = DenseLatentLevel(self.batch_size, encoder_arch, decoder_arch, n_latent, n_det,
    #                                               encoding_form, const_prior_var, variable_update_form, learn_prior)
    #
    #     # construct the output decoder
    #     decoder_arch['n_in'] = self.decoder_input_size(-1, arch)
    #     decoder_arch['n_units'] = arch['n_units_dec'][0]
    #     decoder_arch['n_layers'] = arch['n_layers_dec'][0]
    #     self.output_decoder = MultiLayerPerceptron(**decoder_arch)
    #
    #     # construct the output distribution
    #     if self.output_distribution == 'bernoulli':
    #         self.output_dist = Bernoulli(None)
    #         self.mean_output = Dense(arch['n_units_dec'][0], self.input_size, non_linearity='sigmoid', weight_norm=arch['weight_norm_dec'])
    #     elif self.output_distribution == 'gaussian':
    #         self.output_dist = DiagonalGaussian(None, None)
    #         non_lin = 'linear' if arch['whiten_input'] else 'sigmoid'
    #         self.mean_output = Dense(arch['n_units_dec'][0], self.input_size, non_linearity=non_lin, weight_norm=arch['weight_norm_dec'])
    #         if self.constant_variances:
    #             self.trainable_log_var = Variable(torch.zeros(self.input_size), requires_grad=True)
    #         else:
    #             self.log_var_output = Dense(arch['n_units_dec'][0], self.input_size, weight_norm=arch['weight_norm_dec'])
    #
    # def encoder_input_size(self, level_num, arch):
    #     """Calculates the size of the encoding input to a level."""
    #
    #     def _encoding_size(_self, _level_num, _arch, lower_level=False):
    #
    #         if _level_num == 0:
    #             latent_size = _self.input_size
    #             det_size = 0
    #         else:
    #             latent_size = _arch['n_latent'][_level_num-1]
    #             det_size = _arch['n_det_enc'][_level_num-1]
    #         encoding_size = det_size
    #
    #         if 'posterior' in _self.encoding_form:
    #             encoding_size += latent_size
    #         if 'bottom_error' in _self.encoding_form:
    #             encoding_size += latent_size
    #         if 'bottom_norm_error' in _self.encoding_form:
    #             encoding_size += latent_size
    #         if 'top_error' in _self.encoding_form and not lower_level:
    #             encoding_size += _arch['n_latent'][_level_num]
    #         if 'top_norm_error' in _self.encoding_form and not lower_level:
    #             encoding_size += _arch['n_latent'][_level_num]
    #
    #         return encoding_size
    #
    #     encoder_size = _encoding_size(self, level_num, arch)
    #     if self.concat_variables:
    #         for level in range(level_num):
    #             encoder_size += _encoding_size(self, level, arch, lower_level=True)
    #     return encoder_size
    #
    # def decoder_input_size(self, level_num, arch):
    #     """Calculates the size of the decoding input to a level."""
    #     if level_num == len(arch['n_latent'])-1:
    #         return self.top_size
    #
    #     decoder_size = arch['n_latent'][level_num+1] + arch['n_det_dec'][level_num+1]
    #     if self.concat_variables:
    #         for level in range(level_num+2, len(arch['n_latent'])):
    #             decoder_size += (arch['n_latent'][level] + arch['n_det_dec'][level])
    #     return decoder_size

    def _get_encoding_form(self, observation):
        """
        Gets the appropriate input form for the inference procedure.

        Args:
            observation (Variable, tensor): the input observation
        """
        if self.inference_procedure == 'direct':
            return observation - 0.5
        else:
            raise NotImplementedError

    # def get_input_encoding(self, input):
    #     """Encoding at the bottom level."""
    #     if 'bottom_error' in self.encoding_form or 'bottom_norm_error' in self.encoding_form:
    #         assert self.output_dist is not None, 'Cannot encode error. Output distribution is None.'
    #     encoding = None
    #     if 'posterior' in self.encoding_form:
    #         encoding = input if self.whitening_matrix is not None else input - 0.5
    #     if 'bottom_error' in self.encoding_form:
    #         error = input - self.output_dist.mean.detach()
    #         encoding = error if encoding is None else torch.cat((encoding, error), dim=1)
    #     if 'bottom_norm_error' in self.encoding_form:
    #         error = input - self.output_dist.mean.detach()
    #         norm_error = None
    #         if self.output_distribution == 'gaussian':
    #             norm_error = error / torch.exp(self.output_dist.log_var.detach())
    #         elif self.output_distribution == 'bernoulli':
    #             mean = self.output_dist.mean.detach()
    #             norm_error = error * torch.exp(- torch.log(mean + 1e-5) - torch.log(1 - mean + 1e-5))
    #         encoding = norm_error if encoding is None else torch.cat((encoding, norm_error), dim=1)
    #     return encoding

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        pass

    # def encode(self, input):
    #     """Encodes the input into an updated posterior estimate."""
    #     if self._cuda_device is not None:
    #         input = input.cuda(self._cuda_device)
    #     input = self.process_input(input.view(-1, self.input_size))
    #     h = self.get_input_encoding(input)
    #     for latent_level in self.levels:
    #         if self.concat_variables:
    #             h = torch.cat([h, latent_level.encode(h)], dim=1)
    #         else:
    #             h = latent_level.encode(h)

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        pass

    # def decode(self, generate=False):
    #     """Decodes the posterior (prior) estimate to get a reconstruction (sample)."""
    #     h = Variable(torch.zeros(self.batch_size, self.top_size))
    #     if self._cuda_device is not None:
    #         h = h.cuda(self._cuda_device)
    #     concat = False
    #     for latent_level in self.levels[::-1]:
    #         if self.concat_variables and concat:
    #             h = torch.cat([h, latent_level.decode(h, generate)], dim=1)
    #         else:
    #             h = latent_level.decode(h, generate)
    #         concat = True
    #
    #     h = self.output_decoder(h)
    #     self.output_dist.mean = self.mean_output(h)
    #
    #     if self.output_distribution == 'gaussian':
    #         if self.constant_variances:
    #             self.output_dist.log_var = self.trainable_log_var.unsqueeze(0).repeat(self.batch_size, 1)
    #         else:
    #             self.output_dist.log_var = self.log_var_output(h)
    #
    #     self.process_output(self.output_dist.mean)
    #     return self.output_dist

    def re_init(self, input):
        """
        Method for reinitializing the state (distributions and hidden states).

        Args:
            input (Variable, Tensor): contains observation at t = -1
        """
        pass

    # def reset_state(self):
    #     """Resets the posterior estimate."""
    #     for latent_level in self.levels:
    #         latent_level.reset()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        pass

    # def encoder_parameters(self):
    #     """Returns a list containing all parameters in the encoder."""
    #     params = []
    #     for level in self.levels:
    #         params.extend(level.encoder_parameters())
    #     return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        pass

    # def decoder_parameters(self):
    #     """Returns a list containing all parameters in the decoder."""
    #     params = []
    #     for level in self.levels:
    #         params.extend(level.decoder_parameters())
    #     params.extend(list(self.output_decoder.parameters()))
    #     params.extend(list(self.mean_output.parameters()))
    #     if self.output_distribution == 'gaussian':
    #         if self.constant_variances:
    #             params.append(self.trainable_log_var)
    #         else:
    #             params.extend(list(self.log_var_output.parameters()))
    #     return params

    # def state_parameters(self):
    #     """Returns a list containing the posterior estimate (state)."""
    #     states = []
    #     for latent_level in self.levels:
    #         states.extend(list(latent_level.state_parameters()))
    #     return states

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        pass
