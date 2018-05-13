import pickle
import numpy as np
from torch.utils.data import Dataset


class Basketball(Dataset):

    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.data = pickle.load(open(self.path + '.p', 'rb')).astype('float32')

    def __getitem__(self, index):
        item = self.data[index]
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.data)
