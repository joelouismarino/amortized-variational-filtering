from dlvm import DLVM


class SVG(DLVM):
    """
    Stochastic video generation (SVG) model from "Stochastic Video Generation
    with a Learned Prior," Denton & Fergus, 2018.
    """
    def __init__(self, model_config):
        super(SVG, self).__init__()
