import os
import numpy as np
from torch.utils.data import Dataset


class Blizzard(Dataset):
    """
    A dataset class for the Blizzard dataset.

    Args:
        path (str): path to the directory containing the converted data files
    """
    def __init__(self, path, transforms=None):
        self.path = path
        self.file_list = os.listdir(self.path)
        self.transforms = transforms

    def __getitem__(self, ind):
        f = self.file_list[ind]
        item = np.load(f)
        if self.transforms is not None:
            item = self.transforms(item)
        return item

    def __len__(self):
        return len(self.file_list)
