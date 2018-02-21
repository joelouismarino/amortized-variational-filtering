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
        self.mean_reset_value = Parameter(dt.zeros(1))
        self.log_var_reset_value = Parameter(dt.zeros(1))
        self.mean = mean
        self.log_var = log_var
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draws a tensor of samples.
        :param n_samples: number of samples to draw
        :param resample: whether to resample or just use current sample
        :return: a (batch_size x n_samples x n_variables) tensor of samples
        """
        if self._sample is None or resample:
            mean = self.mean
            std = self.log_var.mul(0.5).exp_()
            if len(self.mean.size()) == 2:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
                std = std.unsqueeze(1).repeat(1, n_samples, 1)
            elif len(self.mean.size()) == 4:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
                std = std.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            rand_normal = Variable(mean.data.new(mean.size()).normal_())
            self._sample = rand_normal.mul_(std).add_(mean)
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
        if len(mean.size()) == 2:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
            log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1)
        elif len(mean.size()) == 4:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)

        # first_term = log_var.mul(-0.5)
        # second_term = np.log(2 * np.pi)
        # third_term_num = (sample.sub(mean)).pow_(2)
        # third_term_den = log_var.exp().add(1e-5)
        # third_term = third_term_num.div_(third_term_den)
        # return first_term.add_(second_term).add_(third_term)
        # first_two_terms = log_var + np.log(2 * np.pi)
        # # second_term = np.log(2 * np.pi)
        # third_term_num = (sample.sub(mean)).pow(2)
        # third_term_den = log_var.exp().add(1e-5)
        # third_term = third_term_num.div(third_term_den)
        # return (first_two_terms.add(third_term)).mul(-0.5)
        # return log_var.mul(-0.5).add_(np.log(2 * np.pi)).add_((sample.sub(mean).pow_(2)).div_(log_var.exp().add(1e-5))) # silent bug

        # new_result = (log_var.add(np.log(2 * np.pi)).add((sample.sub(mean).pow(2)).div(log_var.exp().add(1e-5)))).mul(-0.5)
        # new_result_2 = (log_var.add(np.log(2 * np.pi)).add_((sample.sub(mean).pow_(2)).div_(log_var.exp().add(1e-5)))).mul_(-0.5)
        # old_result = -0.5 * (log_var + np.log(2 * np.pi) + torch.pow(sample - mean, 2) / (torch.exp(log_var) + 1e-5))
        # print new_result.sum().data[0]
        # print new_result_2.sum().data[0]
        # print old_result.sum().data[0]
        # import ipdb; ipdb.set_trace()
        # return old_result
        # return -0.5 * (log_var + np.log(2 * np.pi) + torch.pow(sample - mean, 2) / (torch.exp(log_var) + 1e-5))

        return (log_var.add(np.log(2 * np.pi)).add_((sample.sub(mean).pow_(2)).div_(log_var.exp().add(1e-5)))).mul_(-0.5)

    def reset(self, mean_value=None, log_var_value=None):
        self.reset_mean(mean_value)
        self.reset_log_var(log_var_value)

    def reset_mean(self, value):
        """
        Resets the mean to a particular value.
        :param value: the value to set as the mean, defaults to zero
        :return: None
        """
        mean = value if value is not None else self.mean_reset_value.data
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None

    def reset_log_var(self, value):
        """
        Resets the log variance to a particular value.
        :param value: the value to set as the log variance, defaults to zero
        :return: None
        """
        log_var = value if value is not None else self.log_var_reset_value.data
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

    def parameters(self):
        """
        Gets the distribution parameters.
        :return: tuple of mean and log variance
        """
        return self.mean, self.log_var
