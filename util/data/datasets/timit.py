import numpy as np
import cPickle
from torch.utils.data import Dataset


class TIMIT(Dataset):
    """
    A dataset class for the TIMIT dataset.

    Args:
        path (str): path to the pickle file containing the data list
    """
    def __init__(self, path, transforms=None):
        self.data_list = cPickle.load(open(path, 'r')) # list of audio waveforms
        self.transforms = transforms

    def __getitem__(self, ind):
        item  = self.data_list[ind]
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def __len__(self):
        return len(self.data_list)
