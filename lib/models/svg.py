from latent_variable_model import LatentVariableModel
from lib.modules.latent_levels import LSTMLatentLevel
from lib.modules.networks import LSTMNetwork


class SVG(LatentVariableModel):
    """
    Stochastic video generation (SVG) model from "Stochastic Video Generation
    with a Learned Prior," Denton & Fergus, 2018.

    Args:
        model_config (dict): dictionary containing model configuration params
    """
    def __init__(self, model_config):
        super(SVG, self).__init__(model_config)
        self._construct(model_config)

    def _construct(self, model_config):
        """
        Method for constructing SVG model using the model configuration file.

        Args:
            model_config (dict): dictionary containing model configuration params
        """
        model_type = model_config['model_type'].lower()
        inference_procedure = model_config['inference_procedure'].lower()
        level_config = {}
        latent_config = {}
        latent_config['n_in'] = (256, 256) # number of encoder, decoder units
        latent_config['inference_procedure'] = 'direct' # hard coded because we handle inference here in model
        level_config['inference_procedure'] = 'direct'
        level_config['inference_config'] = {'n_layers': 1, 'n_units': 256, 'n_in': 128}
        level_config['generative_config'] = {'n_layers': 1, 'n_units': 256, 'n_in': 128}
        if model_type == 'sm_mnist':
            from lib.modules.networks.dcgan import encoder, decoder
            self.encoder = encoder(128, 1)
            self.decoder = decoder(128, 1)
            latent_config['n_variables'] = 10
            level_config['latent_config'] = latent_config
        elif model_type == 'kth_actions':
            from lib.modules.networks.vgg16 import encoder, decoder
            self.encoder = encoder(128, 1)
            self.decoder = decoder(128, 2)
            latent_config['n_variables'] = 32
            level_config['latent_config'] = latent_config
        elif model_type == 'bair_robot_pushing':
            from lib.modules.networks.vgg16 import encoder, decoder
            self.encoder = encoder(128, 3)
            self.decoder = decoder(128, 6)
            latent_config['n_variables'] = 64
            level_config['latent_config'] = latent_config
            # if inference_procedure == 'direct':
            #     pass
            # elif inference_procedure == 'iterative':
            #     pass
        else:
            raise Exception('SVG model type must be one of 1) sm_mnist, 2) \
                            kth_action, or 3) bair_robot_pushing. Invalid model \
                            type: ' + model_type + '.')

        self.latent_level = LSTMLatentLevel(level_config)
        self.decoder_lstm = LSTMNetwork({'n_layers': 2, 'n_units': 256,
                                         'n_in': 128 + latent_config['n_variables']})

    def infer(self, observation):
        """
        Method for perfoming inference of the approximate posterior over the
        latent variables.

        Args:
            observation (tensor): observation to infer latent variables from
        """
        h = self.encoder(observation)[0]
        self._h = h
        self.latent_level.infer(h)

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
        g = self.decoder_lstm(torch.cat([z, self._prev_h], dim=2))
        output = self.decoder(g)

    def step(self):
        """
        Method for stepping the generative model forward one step in the sequence.
        """
        self.latent_level.step()
        self.decoder_lstm.step()
        self._prev_h = self._h
        self._h = None

    def re_init(self):
        """
        Method for reinitializing the state (approximate posterior and priors)
        of the dynamical latent variable model.
        """
        self.latent_level.re_init()
        self.decoder_lstm.re_init()
        self._h = self._prev_h = None

    def inference_parameters(self):
        """
        Method for obtaining the inference parameters.
        """
        params = []
        params.extend(list(self.encoder.parameters()))
        params.extend(list(self.latent_level.inference_parameters()))
        return params

    def generative_parameters(self):
        """
        Method for obtaining the generative parameters.
        """
        params = []
        params.extend(list(self.decoder.parameters()))
        params.extend(list(self.latent_level.generative_parameters()))
        params.extend(list(self.decoder_lstm.parameters()))
        return params
