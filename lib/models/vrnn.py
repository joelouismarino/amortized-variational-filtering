from dlvm import DLVM


class VRNN(DLVM):
    """
    Variational recurrent neural network (VRNN) from "A Recurrent Latent
    Variable Model for Sequential Data," Chung et al., 2015.
    """
    def __init__(self, model_config):
        super(VRNN, self).__init__()
