import torch.nn as nn
import torch.distributions.kl as kld


class LatentVariable(nn.Module):
    """
    Abstract class for a latent variable.
    """
    def __init__(self, variable_config):
        super(LatentVariable, self).__init__()
        self.approx_posterior = self.prior = None

    def infer(self, input):
        """
        Abstract method to perform inference.

        Args:
            input (Tensor): input to the inference procedure
        """
        raise NotImplementedError

    def generate(self, input, gen, n_samples=1):
        """
        Abtract method to generate, i.e. run the model forward.

        Args:
            input (Tensor): input to the generative procedure
            gen (boolean): whether to sample from approximate poserior (False) or
                            the prior (True)
            n_samples (int): number of samples to draw
        """
        raise NotImplementedError

    def step(self):
        """
        Abtract method to step the latent variable forward in the sequence.
        """
        raise NotImplementedError

    def re_init(self):
        """
        Abstract method to reinitialize the approximate posterior and prior over
        the variable.
        """
        raise NotImplementedError

    def kl_divergence(self, analytical=False):
        """
        Method to compute KL divergence between the approximate posterior and
        prior for this variable.

        Args:
            analytical (boolean): whether to use the analytical form of the KL
                                  divergence for exact evaluation
        """
        if analytical:
            return kld.kl_divergence(self.approx_posterior, self.prior)
        else:
            z = self.approx_posterior.rsample()
            return self.approx_posterior.log_prob(z) - self.prior.log_prob(z)

    def inference_parameters(self):
        """
        Abstract method to obtain inference parameters.
        """
        raise NotImplementedError

    def generative_parameters(self):
        """
        Abstract method to obtain generative parameters.
        """
        raise NotImplementedError

    def approx_posterior_gradients(self):
        """
        Abstract method to obtain approximate posterior gradients.
        """
        raise NotImplementedError
