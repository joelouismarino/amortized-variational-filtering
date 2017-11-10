import os
import torch
import numpy as np
import skvideo.io as vid
import util.dtypes as dt


class VideoReader(object):
    """
    Converts a video file into a generalized coordinates tensor stream.
    """
    def __init__(self, file_path, n_coords, transforms=None):
        assert os.path.exists(file_path), 'Video file path does not exist.'
        self.frame_generator = vid.vreader(str(file_path))
        self.n_coords = n_coords
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []
        self.buffer = None
        self._cuda_device = None

    def cuda(self, device_id=0):
        # send tensor stream to GPU
        self._cuda_device = device_id
        if self.buffer is not None:
            self.buffer = self.buffer.cuda(self._cuda_device)

    def cpu(self):
        self._cuda_device = None
        if self.buffer is not None:
            self.buffer = self.buffer.cpu()

    def _convert_frame(self, frame):
        # convert the frame to a pytorch tensor, swap dimensions, place on GPU
        frame = np.transpose(frame, (2, 0, 1)).astype('float32')
        frame = torch.from_numpy(frame).type(dt.float)
        return frame

    def _apply_transforms(self, frame):
        # apply (spatial) transformations to the frame
        if 'horizontal_flip' in self.transforms:
            pass
        if 'zca_whiten' in self.transforms:
            pass
        return frame

    def _get_generalized_coords(self, frame):
        # temporal derivative approximation using forward finite differences
        n_channels, height, width = frame.shape
        if self.buffer is None:
            self.buffer = dt.zeros(self.n_coords, n_channels, height, width)
        gen_coords = dt.zeros(self.n_coords, n_channels, height, width)
        gen_coords[0] = frame
        for coord in range(1, self.n_coords):
            gen_coords[coord] = gen_coords[coord-1] - self.buffer[coord-1]
        self.buffer = gen_coords.clone()
        return gen_coords

    def __iter__(self):
        for frame in self.frame_generator:
            frame = self._convert_frame(frame)
            frame = self._apply_transforms(frame)
            yield self._get_generalized_coords(frame)
