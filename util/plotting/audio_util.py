import numpy as np
import torch
try:
    import librosa
except ImportError:
    raise ImportError('Writing audio output requires the librosa library. '+ \
                       'Please install librosa by running pip install librosa')


def write_audio(audio_tensor, audio_write_path, sampling_rate=16000):
    """
    Function to write an audio sample to a file.

    Args:
        audio_tensor (array, Variable, tensor): a tensor containing the audio
                                                sample
        audio_write_path (str): path where the sample is written
        sampling_rate (int): the audio sampling rate (in Hz)
    """
    assert type(audio_write_path) == str, 'Audio write path must be a string.'

    if type(audio_tensor) == torch.autograd.variable.Variable:
        audio_tensor = audio_tensor.data
    if type(audio_tensor) in [torch.cuda.FloatTensor, torch.cuda.IntTensor]:
        audio_tensor = audio_tensor.cpu()
    if type(audio_tensor) in [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor]:
        audio_tensor = audio_tensor.numpy()

    assert type(audio_tensor) == np.ndarray, 'Unknown audio tensor data type.'

    assert len(audio_tensor.shape) == 1, 'Audio sample must have a single dimension.'

    librosa.output.write_wav(audio_write_path, audio_tensor, sampling_rate)
