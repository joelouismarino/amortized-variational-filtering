import math
from torch.nn import Parameter
from torch.autograd import Variable
from distribution import Distribution
import util.dtypes as dt


class Normal(Distribution):
    """
    A normal (Gaussian) density.

    Args:
        mean (tensor): the mean of the distribution
        log_var (tensor): the log variance of the distribution
    """
    def __init__(self, mean=None, log_var=None):
        super(Normal, self).__init__()
        self.mean_reset_value = Parameter(dt.zeros(1))
        self.log_var_reset_value = Parameter(dt.zeros(1))
        self.mean = mean
        self.log_var = log_var
        self._sample = None

    def sample(self, n_samples=1, resample=False):
        """
        Draw samples from the distribution.

        Args:
            n_samples (int): number of samples to draw
            resample (bool): whether to resample or just use current sample

        Return: a tensor of samples
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

    def log_prob(self, value):
        """
        Estimate the log probability, evaluated at the sample.

        Args:
            value (Variable or tensor): the sample to evaluate

        Return: an estimate of log probabilities
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
        return (log_var.add(math.log(2 * math.pi)).add_((sample.sub(mean).pow_(2)).div_(log_var.exp().add(1e-5)))).mul_(-0.5)

    def re_init(self, mean_value=None, log_var_value=None):
        self.re_init_mean(mean_value)
        self.re_init_log_var(log_var_value)

    def re_init_mean(self, value):
        """
        Resets the mean to a particular value.
        :param value: the value to set as the mean, defaults to zero
        :return: None
        """
        mean = value if value is not None else self.mean_reset_value.data.unsqueeze(1)
        self.mean = Variable(mean, requires_grad=True)
        self._sample = None

    def re_init_log_var(self, value):
        """
        Resets the log variance to a particular value.
        :param value: the value to set as the log variance, defaults to zero
        :return: None
        """
        log_var = value if value is not None else self.log_var_reset_value.data.unsqueeze(1)
        self.log_var = Variable(log_var, requires_grad=True)
        self._sample = None
