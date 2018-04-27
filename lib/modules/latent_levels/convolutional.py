import torch.nn as nn
from latent_level import LatentLevel
from lib.modules.networks import ConvolutionalNetwork
from lib.modules.latent_variables import ConvolutionalLatentVariable


class ConvolutionalLatentLevel(LatentLevel):
    """
    Latent level with convolutional encoding and decoding functions.

    Args:
        level_config (dict): dictionary containing level configuration parameters
    """
    def __init__(self, level_config):
        super(ConvolutionalLatentLevel, self).__init__(level_config)
        self._construct(level_config)

    def _construct(self, level_config):
        """
        Method to construct the latent level from the level_config dictionary
        """
        if level_config['inference_config'] is not None:
            self.inference_model = ConvolutionalNetwork(level_config['inference_config'])
        else:
            self.inference_model = lambda x:x
        if level_config['generative_config'] is not None:
            self.generative_model = ConvolutionalNetwork(level_config['generative_config'])
        else:
            self.generative_model = lambda x:x
        self.latent = ConvolutionalLatentVariable(level_config['latent_config'])
        self.inference_procedure = level_config['inference_procedure']

    def _get_encoding_form(self, input):
        """
        Gets the appropriate input form for the inference procedure.
        """
        if self.inference_procedure == 'direct':
            return input
        else:
            raise NotImplementedError

    def infer(self, input):
        """
        Method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        input = self._get_encoding_form(input)
        input = self.inference_model(input)
        self.latent.infer(input)

    def generate(self, input, gen, n_samples):
        """
        Method to generate, i.e. run the model forward.

        Args:
            input (Tensor): input to the generative procedure
            gen (boolean): whether to sample from approximate poserior (False) or
                            the prior (True)
            n_samples (int): number of samples to draw
        """
        if input is not None:
            b, s, c, h, w = input.data.shape
            input = self.generative_model(input.view(-1, c, h, w)).view(b, s, -1, h, w)
        return self.latent.generate(input, gen=gen, n_samples=n_samples)

    def step(self):
        """
        Method to step the latent level forward in the sequence.
        """
        self.latent.step()

    def re_init(self):
        """
        Method to reinitialize the latent level (latent variable and any state
        variables in the generative / inference procedures).
        """
        self.latent.re_init()
        if 're_init' in dir(self.inference_model):
            self.inference_model.re_init()
        if 're_init' in dir(self.generative_model):
            self.generative_model.re_init()

    def inference_parameters(self):
        """
        Method to obtain inference parameters.
        """
        params = nn.ParameterList()
        if 'parameters' in dir(self.inference_model):
            params.extend(list(self.inference_model.parameters()))
        params.extend(list(self.latent.inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method to obtain generative parameters.
        """
        params = nn.ParameterList()
        if 'parameters' in dir(self.generative_model):
            params.extend(list(self.generative_model.parameters()))
        params.extend(list(self.latent.generative_parameters()))
        return params


    # def _get_encoding_form(self, input, in_out):
    #     encoding = input if in_out == 'in' else None
    #     if 'posterior' in self.encoding_form and in_out == 'out':
    #         encoding = input
    #     if ('top_error' in self.encoding_form and in_out == 'in') or ('bottom_error' in self.encoding_form and in_out == 'out'):
    #         error = self.latent.error()
    #         encoding = error if encoding is None else torch.cat((encoding, error), 1)
    #     if 'layer_norm_error' in self.encoding_form:
    #         weighted_error = self.latent.error(weighted=True)
    #         b, c, h, w = weighted_error.data.shape
    #         weighted_error = weighted_error.view(b, c, -1)
    #         weighted_error_mean = weighted_error.mean(dim=2, keepdim=True).view(b, c, 1, 1).repeat(1, 1, h, w)
    #         weighted_error_std = weighted_error.std(dim=2, keepdim=True).view(b, c, 1, 1).repeat(1, 1, h, w)
    #         norm_weighted_error = (weighted_error.view(b, c, h, w) - weighted_error_mean) / (weighted_error_std + 1e-6)
    #         encoding = norm_weighted_error if encoding is None else torch.cat((encoding, norm_weighted_error), 1)
    #     if ('top_weighted_error' in self.encoding_form and in_out == 'in') or ('bottom_weighted_error' in self.encoding_form and in_out == 'out'):
    #         weighted_error = self.latent.error(weighted=True)
    #         encoding = weighted_error if encoding is None else torch.cat((encoding, weighted_error), 1)
    #     if 'mean' in self.encoding_form and in_out == 'in':
    #         approx_post_mean = self.latent.posterior.mean.detach()
    #         if len(approx_post_mean.data.shape) in [3, 5]:
    #             approx_post_mean = approx_post_mean.mean(dim=1)
    #         encoding = approx_post_mean if encoding is None else torch.cat((encoding, approx_post_mean), 1)
    #     if 'log_var' in self.encoding_form and in_out == 'in':
    #         approx_post_log_var = self.latent.posterior.log_var.detach()
    #         if len(approx_post_log_var.data.shape) in [3, 5]:
    #             approx_post_log_var = approx_post_mean.mean(dim=1)
    #         encoding = approx_post_log_var if encoding is None else torch.cat((encoding, approx_post_log_var), 1)
    #     if 'mean_gradient' in self.encoding_form and in_out == 'in':
    #         encoding = self.latent.approx_posterior_gradients()[0] if encoding is None else torch.cat((encoding, self.latent.approx_posterior_gradients()[0]), 1)
    #     if 'log_var_gradient' in self.encoding_form and in_out == 'in':
    #         encoding = self.latent.approx_posterior_gradients()[1] if encoding is None else torch.cat((encoding, self.latent.approx_posterior_gradients()[1]), 1)
    #     if 'layer_norm_gradient' in self.encoding_form and in_out == 'in':
    #         # TODO: this is averaging over the wrong dimensions
    #         mean_grad, log_var_grad = self.state_gradients()
    #         mean_grad_mean = mean_grad.mean(dim=0, keepdim=True)
    #         mean_grad_std = mean_grad.std(dim=0, keepdim=True)
    #         norm_mean_grad = (mean_grad - mean_grad_mean) / (mean_grad_std + 1e-6)
    #         log_var_grad_mean = log_var_grad.mean(dim=0, keepdim=True)
    #         log_var_grad_std = log_var_grad.std(dim=0, keepdim=True)
    #         norm_log_var_grad = (log_var_grad - log_var_grad_mean) / (log_var_grad_std + 1e-6)
    #         norm_grads = [norm_mean_grad, norm_log_var_grad]
    #         encoding = norm_grads if encoding is None else torch.cat((encoding, norm_grads), 1)
    #     return encoding
