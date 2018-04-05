import numpy as np
from torch.utils.data import Dataset


class TIMIT(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.x = data['x']
        self.mask = None
        if 'mask' in data:
            self.mask = data['mask']

    def __get_item__(self, ind):
        return self.x[ind]

    def __len__(self):
        return self.x.shape[0]

# TODO: rewrite so that the dataset loads individual examples and cuts them to
#       a pre-specified length and output dimension, and performs normalization
#
# class TIMIT(Dataset):
#
#     def __init__(self, path, output_dim=200, seq_len=40):
#         pass
#
#     def __get_item__(self, ind):
#         pass
#
#     def __len__(self):
#         pass
