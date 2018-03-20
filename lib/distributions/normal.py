import math
from torch.autograd import Variable
from distribution import Distribution


class Normal(Distribution):
    """
    A normal (Gaussian) density.

    Args:
        mean (tensor): the mean of the distribution
        log_var (tensor): the log variance of the distribution
    """
    def __init__(self, mean=None, log_var=None):
        super(Normal, self).__init__()
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
