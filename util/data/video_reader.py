import os
import torch
import numpy as np
import skvideo.io as vid
import util.dtypes as dt


class VideoReader(object):
    """
    Converts a video file into a tensor stream.
    """
    def __init__(self, file_path, transforms=None):
        assert os.path.exists(file_path), 'Video file path ' + file_path + ' does not exist.'
        self.frame_generator = vid.vreader(str(file_path))
        if transforms is not None:
            self.transforms = transforms
        else:
            self.transforms = []

    def _convert_frame(self, frame):
        # convert the frame to a tensor, swap dimensions
        frame = np.transpose(frame, (2, 0, 1)).astype('float32')
        frame = torch.from_numpy(frame).type(dt.float)
        return frame

    def _apply_transforms(self, frame):
        # apply spatial transformations to the frame
        if 'horizontal_flip' in self.transforms:
            pass
        if 'crop' in self.transforms:
            pass
        if 'resize' in self.transforms:
            pass
        return frame

    def __iter__(self):
        for frame in self.frame_generator:
            frame = self._convert_frame(frame)
            yield self._apply_transforms(frame)
