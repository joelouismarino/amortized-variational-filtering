import torch
from torch.nn import Parameter
from torch.autograd import Variable
from distribution import Distribution
import util.dtypes as dt


class Bernoulli(Distribution):
    """
    A Bernoulli distribution.

    Args:
        mean (tensor): the mean of the distribution
    """
    def __init__(self, mean):
        super(Bernoulli, self).__init__()
        self.mean_reset_value = Parameter(dt.zeros(1))
        self.mean = mean
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draw samples from the distribution.

        Args:
            n_samples (int): number of samples to draw
            resample (bool): whether to resample or just use current sample
        """
        if self._sample is None or resample:
            assert self.mean is not None, 'Mean is None.'
            mean = self.mean
            if len(mean.size()) == 2:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
            elif len(mean.size()) == 4:
                mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            self._sample = torch.bernoulli(mean)
        return self._sample

    def log_prob(self, value):
        """
        Estimate the log probability, evaluated at the value.

        Args:
            value (Variable, tensor, or tuple): the value at which to evaluate
        """
        if value is None:
            value = self.sample()
        assert self.mean is not None, 'Mean is None.'
        n_samples = value.data.shape[1]
        mean = self.mean
        if len(mean.size()) == 2:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1)
            log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1)
        elif len(mean.size()) == 4:
            mean = mean.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
            log_var = log_var.unsqueeze(1).repeat(1, n_samples, 1, 1, 1)
        return value * torch.log(mean + 1e-7) + (1 - value) * torch.log(1 - mean + 1e-7)

    def re_init(self, mean_value=None):
        """
        Re-initializes the distribution.

        Args:
            mean_value (tensor): the value to set as the mean, defaults to zero
        """
        self.re_init_mean(mean_value)

    def re_init_mean(self, value):
        """
        Resets the mean to a particular value.

        Args:
            value (tensor): the value to set as the mean, defaults to zero
        """
        mean = value if value is not None else self.mean_reset_value.data.unsqueeze(1)
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None
