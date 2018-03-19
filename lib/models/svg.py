from latent_variable_model import LatentVariableModel


class SVG(LatentVariableModel):
    """
    Stochastic video generation (SVG) model from "Stochastic Video Generation
    with a Learned Prior," Denton & Fergus, 2018.
    """
    def __init__(self, model_config):
        super(SVG, self).__init__()
