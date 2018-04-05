import torch
import numpy as np
import skvideo.io as vid


def frames_tensor_to_video(frames_tensor, video_write_path):
    """
    Writes a tensor of frames of size [n_frames x n_channels x height x width]
    to a video file at video_write_path.

    Args:
        frames_tensor: PyTorch Variable / tensor or numpy array
        video_write_path: path to write the video
    """
    assert type(video_write_path) == str, 'Video write path must be a string.'

    if type(frames_tensor) == torch.autograd.variable.Variable:
        frames_tensor = frames_tensor.data
    if type(frames_tensor) in [torch.cuda.FloatTensor, torch.cuda.IntTensor]:
        frames_tensor = frames_tensor.cpu()
    if type(frames_tensor) in [torch.FloatTensor, torch.IntTensor, torch.DoubleTensor]:
        frames_tensor = frames_tensor.numpy()

    assert type(frames_tensor) == np.ndarray, 'Unknown frames tensor data type.'

    if np.max(frames_tensor) <= 1:
        frames_tensor = frames_tensor * 255.

    frames_tensor = frames_tensor.transpose(0, 2, 3, 1)
    vid.vwrite(video_write_path, frames_tensor)
