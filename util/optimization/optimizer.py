import torch.optim as opt


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
        self.opt = optimizer(self.parameters, lr=lr)
        self.stored_grads = None
        self.zero_stored_grad()

    def collect(self):
        """
        Collects the current gradients, and adds them to the stored gradients.
        """
        for ind, param in enumerate(self.parameters):
            if self.stored_grads[ind] is None:
                self.stored_grads[ind] = param.grad
            else:
                self.stored_grads[ind] += param.grad

    def step(self):
        """
        Applies the stored gradients to update the parameters.
        """
        for ind, param in enumerate(self.parameters):
            param.grad = self.stored_grads[ind]
        self.opt.step()

    def zero_stored_grad(self):
        """
        Clears the stored gradients.
        """
        self.stored_grads = [None for _ in range(len(list(self.parameters)))]

    def zero_current_grad(self):
        """
        Clears the current gradients.
        """
        self.opt.zero_grad()
