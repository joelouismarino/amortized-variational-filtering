import torch
import torch.nn as nn
from torch.distributions import Normal
from latent_variable import LatentVariable
from ..convolutional import ConvLayer, ConvNetwork


class ConvLatentVariable(LatentVariable):
    """
    A convolutional latent variable. Attributes include a prior, approximate posterior,
    and previous approximate posterior. Methods for generation, inference, and evalution
    of errors, KL-divergence.
    """
    def __init__(self, n_variable_channels, filter_size, const_prior_var, n_input,
                 norm_flow):
        super(ConvolutionalLatentVariable, self).__init__()
        self.n_variable_channels = n_variable_channels
        self.filter_size = filter_size

        self.posterior_mean = Convolutional(n_input[0], self.n_variable_channels, self.filter_size)
        self.posterior_mean_gate = Convolutional(n_input[0], self.n_variable_channels, self.filter_size, 'sigmoid')
        self.posterior_log_var = Convolutional(n_input[0], self.n_variable_channels, self.filter_size)
        self.posterior_log_var_gate = Convolutional(n_input[0], self.n_variable_channels, self.filter_size, 'sigmoid')

        self.prior_mean = Convolutional(n_input[1], self.n_variable_channels, self.filter_size)
        # self.prior_mean_gate = Convolutional(n_input[1], self.n_variable_channels, self.filter_size, 'sigmoid', gate=True)
        self.prior_log_var = None
        if not const_prior_var:
            self.prior_log_var = Convolutional(n_input[1], self.n_variable_channels, self.filter_size)
            # self.prior_log_var_gate = Convolutional(n_input[1], self.n_variable_channels, self.filter_size, 'sigmoid', gate=True)

        self.previous_posterior = Normal(self.n_variable_channels)
        self.posterior = Normal(self.n_variable_channels)
        self.prior = Normal(self.n_variable_channels)
        if const_prior_var:
            self.prior.log_var_trainable()

    def infer(self, input, n_samples=1):
        # infer the approximate posterior
        mean_gate = self.posterior_mean_gate(input)
        mean_update = self.posterior_mean(input) * mean_gate
        # self.posterior.mean = self.posterior.mean.detach() + mean_update
        self.posterior.mean = mean_update
        log_var_gate = self.posterior_log_var_gate(input)
        log_var_update = self.posterior_log_var(input) * log_var_gate
        # self.posterior.log_var = (1. - log_var_gate) * self.posterior.log_var.detach() + log_var_update
        self.posterior.log_var = log_var_update
        return self.posterior.sample(n_samples, resample=True)

    def generate(self, input, gen, n_samples):
        b, s, c, h, w = input.data.shape
        input = input.view(-1, c, h, w)
        # mean_gate = self.prior_mean_gate(input).view(b, s, -1, h, w)
        mean_update = self.prior_mean(input).view(b, s, -1, h, w) # * mean_gate
        # self.prior.mean = (1. - mean_gate) * self.posterior.mean.detach() + mean_update
        self.prior.mean = mean_update
        # log_var_gate = self.prior_log_var_gate(input).view(b, s, -1, h, w)
        log_var_update = self.prior_log_var(input).view(b, s, -1, h, w) # * log_var_gate
        # self.prior.log_var = (1. - log_var_gate) * self.posterior.log_var.detach() + log_var_update
        self.prior.log_var = log_var_update
        if gen:
            return self.prior.sample(n_samples, resample=True)
        return self.posterior.sample(n_samples, resample=True)

    def step(self):
        # set the previous posterior with the current posterior
        self.previous_posterior.mean = self.posterior.mean.detach()
        self.previous_posterior.log_var = self.posterior.log_var.detach()

    def kl_divergence(self, analytical=False):
        if analytical:
            n_samples = self.prior.log_var.data.shape[1]
            post_mean = self.posterior.mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            post_log_var = self.posterior.log_var.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            prior_var = torch.clamp(self.prior.log_var.exp(), 1e-6)
            log_var_ratio = self.prior.log_var - post_log_var
            var_ratio = post_log_var.exp().div(prior_var)
            squared_error = (post_mean - self.prior.mean).pow_(2)
            weighted_squared_error = squared_error.div_(prior_var)
            return (log_var_ratio.add_(var_ratio).add_(weighted_squared_error).sub_(1.)).mul_(0.5)
        else:
            post_log_prob = self.posterior.log_prob(self.posterior.sample())
            prior_log_prob = self.prior.log_prob(self.posterior.sample())
            return post_log_prob.sub(prior_log_prob)

    def error(self, averaged=True, weighted=False):
        sample = self.posterior.sample()
        n_samples = sample.data.shape[1]
        prior_mean = self.prior.mean.detach()
        err = sample - prior_mean[:n_samples]
        if weighted:
            prior_log_var = self.prior.log_var.detach()
            err /= prior_log_var
        if averaged:
            err = err.mean(dim=1)
        return err

    def reset_approx_posterior(self):
        mean = self.prior.mean.data.clone().mean(dim=1)
        log_var = self.prior.log_var.data.clone().mean(dim=1)
        self.posterior.reset(mean, log_var)

    def reset_prior(self):
        self.prior.reset()
        if self.prior_log_var is None:
            self.prior.log_var_trainable()

    def reinitialize_variable(self, output_dims):
        b, _, h, w = output_dims
        # reinitialize the previous approximate posterior and prior
        self.previous_posterior.reset()
        self.previous_posterior.mean = self.previous_posterior.mean.view(1, 1, 1, 1, 1).repeat(b, 1, self.n_variable_channels, h, w)
        self.previous_posterior.log_var = self.previous_posterior.log_var.view(1, 1, 1, 1, 1).repeat(b, 1, self.n_variable_channels, h, w)
        self.prior.reset()
        self.prior.mean = self.prior.mean.view(1, 1, 1, 1, 1).repeat(b, 1, self.n_variable_channels, h, w)
        self.prior.log_var = self.prior.log_var.view(1, 1, 1, 1, 1).repeat(b, 1, self.n_variable_channels, h, w)

    def inference_model_parameters(self):
        inference_params = []
        inference_params.extend(list(self.posterior_mean.parameters()))
        inference_params.extend(list(self.posterior_mean_gate.parameters()))
        inference_params.extend(list(self.posterior_log_var.parameters()))
        inference_params.extend(list(self.posterior_log_var_gate.parameters()))
        return inference_params

    def generative_model_parameters(self):
        generative_params = []
        generative_params.extend(list(self.prior_mean.parameters()))
        if self.prior_log_var is not None:
            generative_params.extend(list(self.prior_log_var.parameters()))
        else:
            generative_params.append(self.prior.log_var)
        return generative_params

    def approx_posterior_parameters(self):
        return [self.posterior.mean.detach(), self.posterior.log_var.detach()]

    def approx_posterior_gradients(self):
        assert self.posterior.mean.grad is not None, 'Approximate posterior gradients are None.'
        grads = [self.posterior.mean.grad.detach()]
        grads += [self.posterior.log_var.grad.detach()]
        for grad in grads:
            grad.volatile = False
        return grads
