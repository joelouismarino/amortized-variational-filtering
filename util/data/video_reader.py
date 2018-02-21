import os
import torch
import numpy as np
from PIL import Image
import skvideo.io as vid
from itertools import izip


class VideoReader(object):
    """
    Converts a list of video files into a batched tensor stream.
    """
    def __init__(self, file_paths, transform=False, resize=(256, 256)):
        """
        file_paths: a list of length batch_size containing paths to video files
        transform: whether to crop/resize the frame_tensor
            Note: this must currently be set to True
        resize: resize dimensions for the frames (h, w)
        """
        self.frame_generators = []
        for file_path in file_paths:
            assert os.path.exists(file_path), 'Video file path ' + file_path + ' does not exist.'
            self.frame_generators.append(vid.vreader(str(file_path)))
        assert transform == True, 'Non-transformed batch not implemented.'
        self.transform = transform
        self.resize = resize

    def _convert_frames(self, frames):
        # convert the list of frames to a tensor, swap dimensions
        if len(frames) > 1:
            frames = np.stack(frames)
        else:
            frames = np.expand_dims(frames[0], axis=0)
        frames = np.transpose(frames, (0, 3, 1, 2)).astype('float32')
        return frames

    def _apply_transforms(self, frame):
        frame = Image.fromarray(frame) # convert to PIL
        # get square center crop and resize
        center = (frame.width / 2., frame.height / 2.)
        half_side = min(frame.width, frame.height) / 2.
        left = max(round(center[0] - half_side), 0)
        right = min(round(center[0] + half_side), frame.width)
        top = max(round(center[1] - half_side), 0)
        bottom = min(round(center[1] + half_side), frame.height)
        frame = frame.crop((left, top, right, bottom))
        frame = frame.resize((self.resize[0], self.resize[1]))
        frame = np.array(frame) # convert back to numpy
        return frame

    def __iter__(self):
        for frames in izip(*self.frame_generators):
            frames = list(frames)
            if self.transform:
                for frame_ind, frame in enumerate(frames):
                    frames[frame_ind] = self._apply_transforms(frame)
            yield self._convert_frames(frames)
