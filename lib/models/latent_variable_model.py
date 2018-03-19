import torch.nn as nn


class LatentVariableModel(nn.Module):
    """
    Abstract class for a (dynamical) latent variable model (DLVM). All models
    inherit from this class.
    """
    def __init__(model_config):
        super(LatentVariableModel, self).__init__()
        self.levels = nn.ModuleList([])
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Abstract method for constructing a dynamical latent variable model using
        the model configuration file.
        """
        raise NotImplementedError

    def infer(self, input):
        """
        Abstract method for perfoming inference of the approximate posterior
        over the latent variables.
        """
        raise NotImplementedError

    def generate(self, gen=False, n_samples=1):
        """
        Abstract method for generating observations, i.e. running the generative
        model forward.
        """
        raise NotImplementedError

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        for latent_level in self.levels:
            latent_level.step()

    def re_init(self):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        for latent_level in self.levels:
            latent_level.re_init()

    def kl_divergences(self, averaged=False):
        """
        Estimate the KL divergence (at each latent level).
        """
        # TODO: keep this general across conv, fc
        kl = []
        for latent_level in self.levels:
            kl.append(latent_level.latent.kl_divergence(analytical=False).sum(4).sum(3).sum(2).mean(1))
        if averaged:
            [level_kl.mean(dim=0) for level_kl in kl]
        else:
            return kl

    def conditional_log_likelihoods(self, input, averaged=False):
        """
        Estimate the conditional log-likelihood.
        """
        # TODO: keep this general across conv, fc
        if len(input.data.shape) == 4:
            input = input.unsqueeze(1) # add sample dimension
        log_prob = self.output_dist.log_prob(sample=input).sub_(np.log(256.))
        log_prob = log_prob.sum(4).sum(3).sum(2).mean(1)
        if averaged:
            log_prob = log_prob.mean(dim=0)
        return log_prob

    def free_energy(self, input, averaged=False):
        """
        Estimate the free energy.
        """
        cond_log_like = self.conditional_log_likelihoods(input)
        kl = sum(self.kl_divergences())
        lower_bound = cond_log_like - kl
        if averaged:
            return lower_bound.mean(dim=0)
        else:
            return lower_bound

    def losses(self, input, averaged=False):
        """
        Estimate all losses.
        """
        cond_log_like = self.conditional_log_likelihoods(input)
        kl = self.kl_divergences()
        lower_bound = cond_log_like - sum(kl)
        if averaged:
            return lower_bound.mean(dim=0), cond_log_like.mean(dim=0), [level_kl.mean(dim=0) for level_kl in kl]
        else:
            return lower_bound, cond_log_like, kl

    def inference_parameters(self):
        pass

    def generative_parameters(self):
        pass
