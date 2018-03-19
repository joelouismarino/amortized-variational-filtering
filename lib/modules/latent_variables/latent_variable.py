import torch.nn as nn


class LatentVariable(nn.Module):
    """
    Abstract class for a latent variable.
    """
    def __init__(self):
        super(LatentVariable, self).__init__()

        self.approx_posterior = None
        self.prior = None

    def infer(self, input):
        """
        Abstract method to perform inference.
        """
        raise NotImplementedError

    def generate(self, input):
        """
        Abtract method to generate, i.e. run the model forward.
        """
        raise NotImplementedError

    def step(self):
        """
        Abtract method to step the latent variable forward in the sequence.
        """
        raise NotImplementedError

    def kl_divergence(self, analytical=False):
        """
        Abtract method to compute KL divergence.
        """
        raise NotImplementedError
