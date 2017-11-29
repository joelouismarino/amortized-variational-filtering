import torch
import torch.nn as nn
from ..distributions.gaussian import DiagonalGaussian
from fully_connected import FullyConnected, FullyConnectedNetwork
from convolutional import Convolutional, ConvolutionalNetwork


class FullyConnectedLatentVariable(nn.Module):
    """
    Fully-connected Gaussian latent variable.
    """
    def __init__(self, n_variables, n_orders_motion, const_prior_var, n_input,
                 norm_flow, learn_prior=True, dynamic=False):
        super(FullyConnectedLatentVariable, self).__init__()
        self.n_variables = n_variables
        self.n_orders_motion = n_orders_motion
        self.learn_prior = learn_prior
        self.dynamic = dynamic

        self.posterior_mean = nn.ModuleList([FullyConnected(n_input[0], self.n_variables) for _ in range(self.n_orders_motion)])
        self.posterior_mean_gate = nn.ModuleList([FullyConnected(n_input[0], self.n_variables, 'sigmoid') for _ in range(self.n_orders_motion)])
        self.posterior_log_var = nn.ModuleList([FullyConnected(n_input[0], self.n_variables) for _ in range(self.n_orders_motion)])
        self.posterior_log_var_gate = nn.ModuleList([FullyConnected(n_input[0], self.n_variables, 'sigmoid') for _ in range(self.n_orders_motion)])

        if self.learn_prior:
            self.prior_mean = nn.ModuleList([FullyConnected(n_input[1], self.n_variables) for _ in range(self.n_orders_motion)])
            self.prior_log_var = None
            if not const_prior_var:
                self.prior_log_var = nn.ModuleList([FullyConnected(n_input[1], self.n_variables) for _ in range(self.n_orders_motion)])

        self.posterior = nn.ModuleList([DiagonalGaussian(self.n_variables) for _ in range(self.n_orders_motion+1)])
        self.prior = nn.ModuleList([DiagonalGaussian(self.n_variables) for _ in range(self.n_orders_motion)])
        if self.learn_prior and const_prior_var:
            self.prior.log_var_trainable()

    def infer(self, input):
        for motion_order in range(self.n_orders_motion):
            mean_update = self.posterior_mean[motion_order](input) * self.posterior_mean_gate[motion_order](input)
            mean = self.posterior[motion_order].mean.detach() + mean_update + self.posterior[motion_order+1].mean
            self.posterior[motion_order].mean = mean
            log_var_update = self.posterior_log_var[motion_order](input) * self.posterior_log_var_gate[motion_order](input)
            log_var = self.posterior[motion_order].log_var.detach() + log_var_update
            self.posterior[motion_order].log_var = log_var
        return torch.cat([p.sample(resample=True) for p in list(self.posterior)[:-1]], dim=2)

    def predict(self, input, generate):
        samples = []
        b, s, n = input.data.shape
        input = input.view(b * s, n)
        for motion_order in range(self.n_orders_motion):
            if self.learn_prior:
                self.prior[motion_order].mean = self.prior_mean[motion_order](input).view(b, s, -1)
                if self.dynamic:
                    self.prior[motion_order].mean += self.posterior[motion_order].mean.detach()
                self.prior[motion_order].log_var = self.prior_log_var[motion_order](input).view(b, s, -1)
        if generate:
            return torch.cat([p.sample(resample=True) for p in self.prior], dim=2)
        return torch.cat([p.sample(resample=True) for p in list(self.posterior)[:-1]], dim=2)

    def kl_divergence(self, analytical=False):
        if analytical:
            pass
        else:
            post_log_prob = torch.cat([post.log_prob(post.sample()) for post in list(self.posterior)[:-1]], dim=2)
            prior_log_prob =  torch.cat([prior.log_prob(post.sample()) for (post, prior) in zip(list(self.posterior)[:-1], self.prior)], dim=2)
            return post_log_prob - prior_log_prob

    def error(self, averaged=True, normalized=False):
        sample = torch.cat([posterior.sample() for posterior in list(self.posterior)[:-1]], dim=2)
        n_samples = sample.data.shape[1]
        prior_mean = torch.cat([prior.mean.detach() for prior in self.prior], dim=2)
        err = sample - prior_mean
        if normalized:
            prior_log_var = torch.cat([prior.log_var.detach() for prior in self.prior], dim=2)
            err /= torch.exp(prior_log_var + 1e-7)
        if averaged:
            err = err.mean(dim=1)
        return err

    def reset(self):
        mean = [prior.mean.data.clone().mean(dim=1) for prior in self.prior]
        log_var = [prior.log_var.data.clone().mean(dim=1) for prior in self.prior]
        for motion_order in range(self.n_orders_motion):
            self.posterior[motion_order].reset(mean[motion_order], log_var[motion_order])

    def inference_parameters(self):
        inference_params = []
        for motion_order in range(self.n_orders_motion):
            inference_params.extend(list(self.posterior_mean[motion_order].parameters()))
            inference_params.extend(list(self.posterior_mean_gate[motion_order].parameters()))
            inference_params.extend(list(self.posterior_log_var[motion_order].parameters()))
            inference_params.extend(list(self.posterior_log_var_gate[motion_order].parameters()))
        return inference_params

    def generative_parameters(self):
        generative_params = []
        if self.learn_prior:
            generative_params.extend(list(self.prior_mean.parameters()))
            if self.prior_log_var is not None:
                generative_params.extend(list(self.prior_log_var.parameters()))
            else:
                generative_params.append(self.prior.log_var)
        return generative_params

    def state_parameters(self):
        state_params = []
        for motion_order in range(self.n_orders_motion):
            state_params.extend(list(self.posterior[motion_order].state_parameters()))
        return state_params

    def state_gradients(self):
        assert self.posterior[0].mean.grad is not None, 'State gradients are None.'
        grads = [torch.cat([posterior.mean.grad.detach() for posterior in list(self.posterior)[:-1]], dim=1)]
        grads += [torch.cat([posterior.log_var.grad.detach() for posterior in list(self.posterior)[:-1]], dim=1)]
        for grad in grads:
            grad.volatile = False
        return grads


class ConvolutionalLatentVariable(nn.Module):

    def __init__(self, n_variable_channels, filter_size, n_orders_motion,
                const_prior_var, n_input, norm_flow, learn_prior=True, dynamic=False):
        super(ConvolutionalLatentVariable, self).__init__()
        self.n_variable_channels = n_variable_channels
        self.filter_size = filter_size
        self.n_orders_motion = n_orders_motion
        self.learn_prior = learn_prior
        self.dynamic = dynamic

        self.posterior_mean = nn.ModuleList([Convolutional(n_input[0], self.n_variable_channels, self.filter_size) for _ in range(self.n_orders_motion)])
        self.posterior_mean_gate = nn.ModuleList([Convolutional(n_input[0], self.n_variable_channels, self.filter_size, 'sigmoid') for _ in range(self.n_orders_motion)])
        self.posterior_log_var = nn.ModuleList([Convolutional(n_input[0], self.n_variable_channels, self.filter_size) for _ in range(self.n_orders_motion)])
        self.posterior_log_var_gate = nn.ModuleList([Convolutional(n_input[0], self.n_variable_channels, self.filter_size, 'sigmoid') for _ in range(self.n_orders_motion)])

        if self.learn_prior:
            self.prior_mean = nn.ModuleList([Convolutional(n_input[1], self.n_variable_channels, self.filter_size) for _ in range(self.n_orders_motion)])
            self.prior_log_var = None
            if not const_prior_var:
                self.prior_log_var = nn.ModuleList([Convolutional(n_input[1], self.n_variable_channels, self.filter_size) for _ in range(self.n_orders_motion)])

        self.posterior = nn.ModuleList([DiagonalGaussian(self.n_variable_channels) for _ in range(self.n_orders_motion+1)])
        self.prior = nn.ModuleList([DiagonalGaussian(self.n_variable_channels) for _ in range(self.n_orders_motion)])
        if self.learn_prior and const_prior_var:
            self.prior.log_var_trainable()

    def infer(self, input):
        for motion_order in range(self.n_orders_motion):
            mean_update = self.posterior_mean[motion_order](input) * self.posterior_mean_gate[motion_order](input)
            mean = self.posterior[motion_order].mean.detach() + mean_update + self.posterior[motion_order + 1].mean
            self.posterior[motion_order].mean = mean
            log_var_update = self.posterior_log_var[motion_order](input) * self.posterior_log_var_gate[motion_order](input)
            log_var = self.posterior[motion_order].log_var.detach() + log_var_update
            self.posterior[motion_order].log_var = log_var
        return torch.cat([p.sample(resample=True) for p in list(self.posterior)[:-1]], dim=2)

    def predict(self, input, generate):
        samples = []
        b, s, c, h, w = input.data.shape
        input = input.view(b * s, c, h, w)
        for motion_order in range(self.n_orders_motion):
            if self.learn_prior:
                self.prior[motion_order].mean = self.prior_mean[motion_order](input).view(b, s, -1, h, w)
                if self.dynamic:
                    self.prior[motion_order].mean += self.posterior[motion_order].mean.detach()
                self.prior[motion_order].log_var = self.prior_log_var[motion_order](input).view(b, s, -1, h, w)
        if generate:
            return torch.cat([p.sample(resample=True) for p in self.prior], dim=2)
        return torch.cat([p.sample(resample=True) for p in list(self.posterior)[:-1]], dim=2)

    def kl_divergence(self, analytical=False):
        if analytical:
            pass
        else:
            post_log_prob = torch.cat([post.log_prob(post.sample()) for post in list(self.posterior)[:-1]], dim=2)
            prior_log_prob = torch.cat([prior.log_prob(post.sample()) for (post, prior) in zip(list(self.posterior)[:-1], self.prior)], dim=2)
            return post_log_prob - prior_log_prob

    def error(self, averaged=True, normalized=False):
        sample = torch.cat([posterior.sample() for posterior in list(self.posterior)[:-1]], dim=2)
        n_samples = sample.data.shape[1]
        prior_mean = torch.cat([prior.mean.detach() for prior in self.prior], dim=2)
        err = sample - prior_mean
        if normalized:
            prior_log_var = torch.cat([prior.log_var.detach() for prior in self.prior], dim=2)
            err /= prior_log_var
        if averaged:
            err = err.mean(dim=1)
        return err

    def reset(self):
        mean = [prior.mean.data.clone().mean(dim=1) for prior in self.prior]
        log_var = [prior.log_var.data.clone().mean(dim=1) for prior in self.prior]
        for motion_order in range(self.n_orders_motion):
            self.posterior[motion_order].reset(mean[motion_order], log_var[motion_order])

    def inference_parameters(self):
        inference_params = []
        for motion_order in range(self.n_orders_motion):
            inference_params.extend(list(self.posterior_mean[motion_order].parameters()))
            inference_params.extend(list(self.posterior_mean_gate[motion_order].parameters()))
            inference_params.extend(list(self.posterior_log_var[motion_order].parameters()))
            inference_params.extend(list(self.posterior_log_var_gate[motion_order].parameters()))
        return inference_params

    def generative_parameters(self):
        generative_params = []
        if self.learn_prior:
            generative_params.extend(list(self.prior_mean.parameters()))
            if self.prior_log_var is not None:
                generative_params.extend(list(self.prior_log_var.parameters()))
            else:
                generative_params.append(self.prior.log_var)
        return generative_params

    def state_parameters(self):
        state_params = []
        for motion_order in range(self.n_orders_motion):
            state_params.extend(list(self.posterior[motion_order].state_parameters()))
        return state_params

    def state_gradients(self):
        assert self.posterior[0].mean.grad is not None, 'State gradients are None.'
        grads = [torch.cat([posterior.mean.grad.detach() for posterior in list(self.posterior)[:-1]], dim=1)]
        grads += [torch.cat([posterior.log_var.grad.detach() for posterior in list(self.posterior)[:-1]], dim=1)]
        for grad in grads:
            grad.volatile = False
        return grads
