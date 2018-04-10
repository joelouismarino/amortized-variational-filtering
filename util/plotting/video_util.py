import torch
import numpy as np
import skvideo.io as vid
import matplotlib.pyplot as plt


def convert_tensor(frames_tensor):
    """
    Converts a Variable or tensor of video frames into a format that is able to
    be visualized and written.

    Args:
        frames_tensor (PyTorch tensor / Variable or numpy array): tensor of frames
                                                        expected to be of size
                                                        [n, c, h, w], where n
                                                        is the number of frames
    """
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
    return frames_tensor


def frames_tensor_to_video(frames_tensor, video_write_path):
    """
    Writes a tensor of frames of size [n_frames x n_channels x height x width]
    to a video file at video_write_path.

    Args:
        frames_tensor (PyTorch Variable / tensor or numpy array): tensor of frames
        video_write_path (str): path to write the video
    """
    assert type(video_write_path) == str, 'Video write path must be a string.'
    frames_tensor = convert_tensor(frames_tensor)
    vid.vwrite(video_write_path, frames_tensor)


def plot_data_and_recon(data_tensor, recon_tensor):
    """
    Plots the reconstruction of a video sequence alongside the original sequence.

    Args:
        data_tensor (PyTorch Variable / tensor or numpy array): original data
        recon_tensor(PyTorch Variable / tensor or numpy array): reconstruction
    """
    data_tensor = convert_tensor(data_tensor)
    recon_tensor = convert_tensor(recon_tensor)
    n_frames = data_tensor.shape[0]

    for frame_num in range(1, n_frames+1):
        # plot the data
        plt.subplot(2, n_frames, frame_num)
        plt.imshow(data_tensor[frame_num-1].astype('uint8'))
        plt.axis('off')
        plt.title('t = ' + str(frame_num))

        # plot the reconstruction
        plt.subplot(2, n_frames, frame_num + n_frames)
        plt.imshow(recon_tensor[frame_num-1].astype('uint8'))
        plt.axis('off')

    plt.show()
