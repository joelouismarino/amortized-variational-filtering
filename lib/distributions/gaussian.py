import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.autograd import Variable
import util.dtypes as dt
import numpy as np


class DiagonalGaussian(nn.Module):
    """
    A diagonal Gaussian density.
    """
    def __init__(self, n_variables, mean=None, log_var=None):
        super(DiagonalGaussian, self).__init__()
        self.n_variables = n_variables
        self.mean = mean if mean is not None else Variable(dt.zeros(1))
        self.log_var = log_var if log_var is not None else Variable(dt.zeros(1))
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draws a tensor of samples.
        :param n_samples: number of samples to draw
        :param resample: whether to resample or just use current sample
        :return: a (batch_size x n_samples x n_variables) tensor of samples
        """
        if self._sample is None or resample:
            mean_shape = list(self.mean.data.shape)
            if len(mean_shape) in [2, 4]:
                mean_shape.insert(1, n_samples)
            rand_normal = Variable(torch.randn(tuple(mean_shape)).type(dt.float))
            mean = self.mean
            log_var = self.log_var
            if len(mean_shape) == 2:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
                log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1)
            elif len(mean_shape) == 4:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
                log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            self._sample = mean + torch.exp(0.5 * log_var) * rand_normal
        return self._sample

    def log_prob(self, sample=None):
        """
        Estimates the log probability, evaluated at the sample.
        :param sample: the sample to evaluate log probability at
        :return: a (batch_size x n_samples x n_variables) estimate of log probabilities
        """
        if sample is None:
            sample = self.sample()
        assert self.mean is not None and self.log_var is not None, 'Mean or log variance are None.'
        n_samples = sample.data.shape[1]
        mean = self.mean
        log_var = self.log_var
        if len(mean.data.shape) == 2:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
            log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1)
        elif len(mean.data.shape) == 4:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
        return -0.5 * (log_var + np.log(2 * np.pi) + torch.pow(sample - mean, 2) / (torch.exp(log_var) + 1e-5))

    def reset(self, mean_value, log_var_value):
        self.reset_mean(mean_value)
        self.reset_log_var(log_var_value)

    def reset_mean(self, value=None):
        """
        Resets the mean to a particular value.
        :param value: the value to set as the mean, defaults to zero
        :return: None
        """
        assert self.mean is not None or value is not None, 'Mean is None.'
        mean = value if value is not None else dt.zeros(self.mean.size())
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None

    def reset_log_var(self, value=None):
        """
        Resets the log variance to a particular value.
        :param value: the value to set as the log variance, defaults to zero
        :return: None
        """
        assert self.log_var is not None or value is not None, 'Log variance is None.'
        log_var = value if value is not None else dt.zeros(self.log_var.size())
        self.log_var = Variable(log_var, requires_grad=True)
        self._sample = None

    def mean_trainable(self):
        """
        Makes the mean a trainable variable.
        :return: None
        """
        assert self.mean is not None, 'Mean is None.'
        self.mean = Variable(self.mean.data.clone(), requires_grad=True)

    def log_var_trainable(self):
        """
        Makes the log variance a trainable variable.
        :return: None
        """
        assert self.log_var is not None, 'Log variance is None.'
        self.log_var = Variable(self.log_var.data.clone(), requires_grad=True)

    def mean_not_trainable(self):
        """
        Makes the mean a non-trainable variable.
        :return: None
        """
        self.mean.requires_grad = False

    def log_var_not_trainable(self):
        """
        Makes the log variance a non-trainable variable.
        :return: None
        """
        self.log_var.requires_grad = False

    def state_parameters(self):
        """
        Gets the state parameters for this variable.
        :return: tuple of mean and log variance
        """
        return self.mean, self.log_var
