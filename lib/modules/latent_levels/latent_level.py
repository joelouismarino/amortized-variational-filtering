import torch.nn as nn


class LatentLevel(nn.Module):
    """
    Abstract class for a latent level.
    """
    def __init__(self):
        super(LatentLevel, self).__init__()

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

    def inference_parameters(self):
        """
        Returns the inference parameters.
        """
        inference_params = []
        inference_params.extend(list(self.encoder.parameters()))
        if self.det_enc:
            inference_params.extend(list(self.det_enc.parameters()))
        inference_params.extend(list(self.latent.inference_model_parameters()))
        return inference_params

    def generative_parameters(self):
        """
        Returns the generative parameters.
        """
        generative_params = []
        generative_params.extend(list(self.decoder.parameters()))
        if self.det_dec:
            generative_params.extend(list(self.det_dec.parameters()))
        generative_params.extend(list(self.latent.generative_model_parameters()))
        return generative_params
