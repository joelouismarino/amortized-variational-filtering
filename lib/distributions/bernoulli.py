import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class Bernoulli(object):

    def __init__(self, n_variables, mean=None):
        """
        Creates a Bernoulli distribution.
        :param n_variables: the size (number of dimensions) of the distribution.
        :param mean: mean of the Bernoulli distribution.
        """
        self.n_variables = n_variables
        self.mean = mean
        self._sample = None
        self._cuda_device = None

    def sample(self, n_samples=1, resample=False):
        """
        Draws a tensor of samples.
        :param n_samples: number of samples to draw
        :param resample: whether to resample or just use current sample
        :return: a (batch_size x n_samples x n_variables) tensor of samples
        """
        if self._sample is None or resample:
            assert self.mean is not None, 'Mean is None.'
            mean = self.mean.unsqueeze(1).repeat(1, n_samples, 1)
            self._sample = torch.bernoulli(mean)
        return self._sample

    def log_prob(self, sample=None):
        """
        Estimates the log probability, evaluated at the sample.
        :param sample: the sample to evaluate log probability at
        :return: a (batch_size x n_samples x n_variables) estimate of log probabilities
        """
        if sample is None:
            sample = self.sample()
        assert self.mean is not None, 'Mean is None.'
        n_samples = sample.size()[1]
        if len(self.mean.data.shape) == 2:
            mean = self.mean.unsqueeze(1).repeat(1, n_samples, 1)
        else:
            mean = self.mean
        return sample * torch.log(mean + 1e-7) + (1 - sample) * torch.log(1 - mean + 1e-7)

    def reset_mean(self, value=None):
        """
        Resets the mean to a particular value.
        :param value: the value to set as the mean, defaults to zero
        :return: None
        """
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else torch.zeros(self.mean.size())
        if self._cuda_device is not None:
            mean = mean.cuda(self._cuda_device)
        mean = Variable(mean, requires_grad=self.mean.requires_grad)
        self.mean = mean
        self._sample = None

    def mean_trainable(self):
        """
        Makes the mean a trainable variable.
        :return: None
        """
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data, requires_grad=True)

    def mean_not_trainable(self):
        """
        Makes the mean a non-trainable variable.
        :return: None
        """
        self.mean.requires_grad = False

    def state_parameters(self):
        """
        Gets the state parameters for this variable.
        :return: tuple of mean and log variance
        """
        return self.mean

    def cuda(self, device_id):
        """
        Places the distribution on the GPU.
        :param device_id: device on which to place the distribution
        :return: None
        """
        if self.mean is not None:
            self.mean = Variable(self.mean.data.cuda(device_id), requires_grad=self.mean.requires_grad)
        self._cuda_device = device_id

    def cpu(self):
        """
        Places the distribution on the CPU.
        :return: None
        """
        if self.mean is not None:
            self.mean = self.mean.cpu()
        self._cuda_device = None
