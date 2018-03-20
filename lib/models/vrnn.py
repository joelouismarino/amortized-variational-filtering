import torch
from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import FullyConnectedLatentLevel
from lib.modules.networks import LSTMNetwork


class VRNN(LatentVariableModel):
    """
    Variational recurrent neural network (VRNN) from "A Recurrent Latent
    Variable Model for Sequential Data," Chung et al., 2015.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(VRNN, self).__init__()
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Args:
            model_config (dict): dictionary containing model configuration params
        """
        model_type = model_config['model_type'].lower()
        if model_type == 'timit':
            self.lstm = LSTMNetwork(1, 2000)
            self.x_model = FullyConnectedNetwork(4, 600)
            self.z_model = FullyConnectedNetwork(4, 600)
            self.encoder_model = FullyConnectedNetwork(4, 600)
            self.prior_model = FullyConnectedNetwork(4, 600)
            self.decoder_model = FullyConnectedNetwork(4, 600)
        elif model_type == 'blizzard':
            self.lstm = LSTMNetwork(1, 4000)
            self.x_model = FullyConnectedNetwork(4, 800)
            self.z_model = FullyConnectedNetwork(4, 800)
            self.encoder_model = FullyConnectedNetwork(4, 800)
            self.prior_model = FullyConnectedNetwork(4, 800)
            self.decoder_model = FullyConnectedNetwork(4, 800)
        elif model_type == 'iam_ondb':
            self.lstm = LSTMNetwork(1, 1200)
            self.x_model = FullyConnectedNetwork(4, 600)
            self.z_model = FullyConnectedNetwork(4, 600)
            self.encoder_model = FullyConnectedNetwork(4, 600)
            self.prior_model = FullyConnectedNetwork(4, 600)
            self.decoder_model = FullyConnectedNetwork(4, 600)
        else:
            raise Exception('VRNN model type must be one of 1) timit, 2) \
                            blizzard, or 3) iam_ondb. Invalid model \
                            type: ' + model_type + '.')

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        self._x_enc = self.x_model(observation)
        h_enc = self.encoder_model(torch.cat([self._x_enc, self._h], dim=1))
        self.latent.infer(h_enc)

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        z = self.latent.generate(self.prior_model(self._h), gen=gen, n_samples=n_samples)
        self._z_enc = self.z_model(z)
        output = self.decoder_model(torch.cat([self._z_enc, self._h], dim=2))

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        self._h = self.lstm(torch.cat([self._x_enc, self._z_enc[:, 0]], dim=1)

    def re_init(self):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        self.latent.re_init()
        self.lstm.re_init()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        pass

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        pass
