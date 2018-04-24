import torch
import torch.optim as opt
from torch.autograd import Variable


class Optimizer(object):
    """
    Wrapper class for PyTorch optimizer to store gradients.

    Args:
        opt_name (str): name of the optimizer
        parameters (iterable): the parameters to optimize
        lr (float): the learning rate for the optimizer
    """
    def __init__(self, opt_name, parameters, lr):
        opt_name = opt_name.lower().replace('_', '').strip()
        if opt_name == 'sgd':
            optimizer = opt.SGD
        elif opt_name == 'rmsprop':
            optimizer = opt.RMSprop
        elif opt_name == 'adam':
            optimizer = opt.Adam
        self.parameters = list(parameters)
        if self.parameters == []:
            # in case we're not using the optimizer
            self.parameters = [Variable(torch.zeros(1), requires_grad=True)]
        self.opt = optimizer(self.parameters, lr=lr)
        self.stored_grads = None
        self.zero_stored_grad()
        self._n_iter = 0

    def collect(self):
        """
        Collects the current gradients, and adds them to the stored gradients.
        """
        for ind, param in enumerate(self.parameters):
            if param.grad is not None:
                self.stored_grads[ind] += param.grad

    def step(self):
        """
        Applies the stored gradients to update the parameters.
        """
        assert self._n_iter > 0, 'The optimizer does not have gradients to apply.'
        for ind, param in enumerate(self.parameters):
            param.grad = self.stored_grads[ind] / self._n_iter
        self.opt.step()
        self._n_iter = 0
        self.zero_stored_grad()
        self.zero_current_grad()

    def step_iter(self, n_steps=1):
        """
        Steps the internal iterator that is used to scale the gradients.

        Args:
            n_steps (optional, int): number of steps to step the iterator
        """
        self._n_iter += n_steps

    def zero_stored_grad(self):
        """
        Replaces the stored gradients with zeros.
        """
        self.stored_grads = [Variable(param.data.new(param.size()).zero_()) \
                            for param in self.parameters]

    def zero_current_grad(self):
        """
        Clears the current gradients.
        """
        self.opt.zero_grad()
