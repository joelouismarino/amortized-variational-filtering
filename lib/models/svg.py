from latent_variable_model import LatentVariableModel
from lib.modules.networks import LSTMNetwork


class SVG(LatentVariableModel):
    """
    Stochastic video generation (SVG) model from "Stochastic Video Generation
    with a Learned Prior," Denton & Fergus, 2018.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(SVG, self).__init__()
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Method for constructing SVG model using the model configuration file.

        Args:
            model_config (dict): dictionary containing model configuration params
        """
        # TODO: roll the LSTMs and latent variables into latent levels
        model_type = model_config['model_type'].lower()
        if model_type == 'sm_mnist':
            from lib.modules.networks.dcgan import encoder, decoder
            self.encoder = encoder(128, 1)
            self.decoder = decoder(128, 1)
            self.latent = FullyConnectedLatentVariable(10)
        elif model_type == 'kth_actions':
            from lib.modules.networks.vgg16 import encoder, decoder
            self.encoder = encoder(128, 1)
            self.decoder = decoder(128, 1)
            self.latent = FullyConnectedLatentVariable(32)
        elif model_type == 'bair_robot_pushing':
            from lib.modules.networks.vgg16 import encoder, decoder
            self.encoder = encoder(128, 3)
            self.decoder = decoder(128, 3)
            self.latent = FullyConnectedLatentVariable(64)
        else:
            raise Exception('SVG model type must be one of 1) sm_mnist, 2) \
                            kth_action, or 3) bair_robot_pushing. Invalid model \
                            type: ' + model_type + '.')

        # self.inference_lstm = LSTMNetwork(1, 256)
        # self.prior_lstm = LSTMNetwork(1, 256)
        # self.cond_like_lstm = LSTMNetwork(2, 256)

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        h = self.encoder(observation)
        self._h = h
        self.latent_level.infer(h)
        # self._h = h
        # h = self.inference_lstm(h)
        # self.latent.infer(h)

    def generate(self, gen=False, n_samples=1):
        """
        Method for generating observations, i.e. running the generative model
        forward.

        Args:
            gen (boolean): whether to sample from prior or approximate posterior
            n_samples (int): number of samples to draw and evaluate
        """
        # generate the prior, sample from the latent variables
        z = self.latent_level.generate(self._prev_h, gen=gen, n_samples=n_samples)
        g = self.cond_like_lstm(z)
        output = self.decoder(torch.cat([g, self._prev_h], dim=2))

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        self._prev_h = self._h
        self._h = None

    def re_init(self):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        self.latent_level.re_init()

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = []
        params.extend(self.encoder.parameters())
        params.extend(self.latent_level.inference_parameters())
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = []
        params.extend(self.decoder.parameters())
        params.extend(self.latent_level.generative_parameters)
        params.extend(self.cond_like_lstm.parameters())
        return params
