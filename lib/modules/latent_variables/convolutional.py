import torch
import torch.nn as nn
from lib.distributions import Normal
from latent_variable import LatentVariable
from lib.modules.layers import ConvolutionalLayer
from lib.modules.misc import LayerNorm


class ConvolutionalLatentVariable(LatentVariable):
    """
    A convolutional (Gaussian) latent variable.

    Args:
        latent_config (dict): dictionary containing variable config parameters
    """
    def __init__(self, latent_config):
        super(ConvolutionalLatentVariable, self).__init__(latent_config)
        self._construct(latent_config)

    def _construct(self, latent_config):
        """
        Constructs the latent variable according to the latent_config dict.

        Args:
            latent_config (dict): dictionary containing variable config params
        """
        self.inference_procedure = latent_config['inference_procedure']
        if self.inference_procedure in ['gradient', 'error']:
            self.update_type = latent_config['update_type']
        n_channels = latent_config['n_channels']
        filter_size = latent_config['filter_size']
        n_inputs = latent_config['n_in']

        self.normalize_samples = latent_config['normalize_samples']
        if self.normalize_samples:
            self.normalizer = LayerNorm()

        if self.inference_procedure in ['direct', 'gradient', 'error']:
            # approximate posterior inputs
            self.inf_mean_output = ConvolutionalLayer({'n_in': n_inputs[0],
                                                       'n_filters': n_channels,
                                                       'filter_size': filter_size})
            self.inf_log_var_output = ConvolutionalLayer({'n_in': n_inputs[0],
                                                          'n_filters': n_channels,
                                                          'filter_size': filter_size})

        if self.inference_procedure in ['gradient', 'error']:
            self.approx_post_mean_gate = ConvolutionalLayer({'n_in': n_inputs[0],
                                                             'n_filters': n_channels,
                                                             'filter_size': filter_size,
                                                             'non_linearity': 'sigmoid'})
            self.approx_post_log_var_gate = ConvolutionalLayer({'n_in': n_inputs[0],
                                                                'n_filters': n_channels,
                                                                'filter_size': filter_size,
                                                                'non_linearity': 'sigmoid'})

        # prior inputs
        self.prior_mean = ConvolutionalLayer({'n_in': n_inputs[1],
                                              'n_filters': n_channels,
                                              'filter_size': filter_size})
        self.prior_log_var = ConvolutionalLayer({'n_in': n_inputs[1],
                                                 'n_filters': n_channels,
                                                 'filter_size': filter_size})

        # distributions
        self.approx_post = Normal()
        self.prior = Normal()
        self.approx_post.re_init()
        self.prior.re_init()

    def infer(self, input):
        """
        Method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        if self.inference_procedure in ['direct', 'gradient', 'error']:
            approx_post_mean = self.inf_mean_output(input)
            approx_post_log_var = self.inf_log_var_output(input)

        if self.inference_procedure == 'direct':
            self.approx_post.mean = approx_post_mean
            self.approx_post.log_var = torch.clamp(approx_post_log_var, -15, 5)
        elif self.inference_procedure in ['gradient', 'error']:
            if self.update_type == 'highway':
                # gated highway update
                approx_post_mean_gate = self.approx_post_mean_gate(input)
                self.approx_post.mean = approx_post_mean_gate * self.approx_post.mean.detach() \
                                        + (1 - approx_post_mean_gate) * approx_post_mean
                approx_post_log_var_gate = self.approx_post_log_var_gate(input)
                self.approx_post.log_var = torch.clamp(approx_post_log_var_gate * self.approx_post.log_var.detach() \
                                           + (1 - approx_post_log_var_gate) * approx_post_log_var, -15, 5)
        elif self.update_type == 'learned_sgd':
            # SGD style update with learned learning rate and offset
            mean_grad, log_var_grad = self.approx_posterior_gradients()
            mean_lr = self.approx_post_mean_gate(input)
            log_var_lr = self.approx_post_log_var_gate(input)
            self.approx_post.mean = self.approx_post.mean.detach() - mean_lr * mean_grad + approx_post_mean
            self.approx_post.log_var = torch.clamp(self.approx_post.log_var.detach() - log_var_lr * log_var_grad + approx_post_log_var, -15, 5)
        else:
            raise NotImplementedError

        if self.normalize_samples:
            # apply layer normalization to the approximate posterior means
            self.approx_post.mean = self.normalizer(self.approx_post.mean)

        # retain the gradients (for inference)
        self.approx_post.mean.retain_grad()
        self.approx_post.log_var.retain_grad()

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
            input = input.view(b * s, c, h, w)
            self.prior.mean = self.prior_mean(input).view(b, s, -1, h, w)
            self.prior.log_var = torch.clamp(self.prior_log_var(input).view(b, s, -1, h, w), -15, 5)
        dist = self.prior if gen else self.approx_post
        sample = dist.sample(n_samples, resample=True)
        sample = sample.detach() if self.detach else sample
        return sample

    def re_init(self):
        """
        Method to reinitialize the approximate posterior and prior over the variable.
        """
        # TODO: this is wrong. we shouldnt set the posterior to the prior then zero out the prior...
        self.re_init_approx_posterior()
        self.prior.re_init()

    def re_init_approx_posterior(self):
        """
        Method to reinitialize the approximate posterior.
        """
        mean = self.prior.mean.detach().mean(dim=1).data
        log_var = self.prior.log_var.detach().mean(dim=1).data
        self.approx_post.re_init(mean, log_var)

    def step(self):
        """
        Method to step the latent variable forward in the sequence.
        """
        pass

    def error(self, averaged=True):
        """
        Calculates Gaussian error for encoding.

        Args:
            averaged (boolean): whether or not to average over samples
        """
        sample = self.approx_post.sample()
        n_samples = sample.data.shape[1]
        prior_mean = self.prior.mean.detach()
        if len(prior_mean.data.shape) == 2:
            prior_mean = prior_mean.unsqueeze(1).repeat(1, n_samples, 1)
        prior_log_var = self.prior.log_var.detach()
        if len(prior_log_var.data.shape) == 2:
            prior_log_var = prior_log_var.unsqueeze(1).repeat(1, n_samples, 1)
        n_error = (sample - prior_mean) / torch.exp(prior_log_var + 1e-7)
        if averaged:
            n_error = n_error.mean(dim=1)
        return n_error

    def close_gates(self):
        nn.init.constant(self.approx_post_mean_gate.linear.bias, 5.)
        nn.init.constant(self.approx_post_log_var_gate.linear.bias, 5.)

    def inference_parameters(self):
        """
        Method to obtain inference parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.inf_mean_output.parameters()))
        params.extend(list(self.inf_log_var_output.parameters()))
        # params.extend(list(self.approx_post_mean.parameters()))
        # params.extend(list(self.approx_post_log_var.parameters()))
        if self.inference_procedure != 'direct':
            params.extend(list(self.approx_post_mean_gate.parameters()))
            params.extend(list(self.approx_post_log_var_gate.parameters()))
        return params

    def generative_parameters(self):
        """
        Method to obtain generative parameters.
        """
        params = nn.ParameterList()
        params.extend(list(self.prior_mean.parameters()))
        params.extend(list(self.prior_log_var.parameters()))
        return params

    def approx_posterior_parameters(self):
        return [self.approx_post.mean.detach(), self.approx_post.log_var.detach()]

    def approx_posterior_gradients(self):
        assert self.approx_post.mean.grad is not None, 'Approximate posterior gradients are None.'
        grads = [self.approx_post.mean.grad.detach()]
        grads += [self.approx_post.log_var.grad.detach()]
        for grad in grads:
            grad.volatile = False
        return grads
