import torch
import torch.nn as nn
from torch.distributions import Normal
from latent_variable import LatentVariable
from fully_connected import FullyConnectedLayer, FullyConnectedNetwork


class FullyConnectedLatentVariable(LatentVariable):
    """
    A fully-connected latent variable. Attributes include a prior, approximate posterior,
    and previous approximate posterior. Methods for generation, inference, and evalution
    of errors, KL-divergence.
    """
    def __init__(self, n_variables, const_prior_var, n_input, norm_flow):
        super(FullyConnectedLatentVariable, self).__init__()
        self.n_variables = n_variables

        self.posterior_mean = FullyConnected(n_input[0], self.n_variables)
        self.posterior_mean_gate = FullyConnected(n_input[0], self.n_variables, 'sigmoid')
        self.posterior_log_var = FullyConnected(n_input[0], self.n_variables)
        self.posterior_log_var_gate = FullyConnected(n_input[0], self.n_variables, 'sigmoid')

        self.prior_mean = FullyConnected(n_input[1], self.n_variables)
        self.prior_log_var = None
        if not const_prior_var:
            self.prior_log_var = FullyConnected(n_input[1], self.n_variables)

        self.previous_posterior = DiagonalGaussian(self.n_variables)
        self.posterior = DiagonalGaussian(self.n_variables)
        self.prior = DiagonalGaussian(self.n_variables)
        if const_prior_var:
            self.prior.log_var_trainable()

    def infer(self, input, n_samples):
        # infer the approximate posterior
        mean_update = self.posterior_mean(input) * self.posterior_mean_gate(input)
        self.posterior.mean = self.posterior.mean.detach() + mean_update
        log_var_update = self.posterior_log_var(input) * self.posterior_log_var_gate(input)
        self.posterior.log_var = self.posterior.log_var.detach() + log_var_update
        return self.posterior.sample(n_samples, resample=True)

    def generate(self, input, n_samples, gen):
        b, s, n = input.data.shape
        input = input.view(b * s, n)
        self.prior.mean = self.prior_mean(input).view(b, s, -1)
        self.prior.log_var = self.prior_log_var(input).view(b, s, -1)
        if gen:
            return self.prior.sample(n_samples, resample=True)
        return self.posterior.sample(n_samples, resample=True)

    def kl_divergence(self, analytical=False):
        if analytical:
            pass
        else:
            post_log_prob = self.posterior.log_prob(self.posterior.sample())
            prior_log_prob =  self.prior.log_prob(self.posterior.sample())
            return post_log_prob - prior_log_prob

    def error(self, averaged=True, normalized=False):
        sample = self.posterior.sample()
        n_samples = sample.data.shape[1]
        prior_mean = self.prior.mean.detach()
        err = sample - prior_mean[:n_samples]
        if normalized:
            prior_log_var = self.prior.log_var.detach()
            err /= torch.exp(prior_log_var + 1e-7)
        if averaged:
            err = err.mean(dim=1)
        return err

    def reset(self):
        mean = self.prior.mean.data.clone().mean(dim=1)
        log_var = self.prior.log_var.data.clone().mean(dim=1)
        self.posterior.reset(mean, log_var)

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
