from torch.utils.data import DataLoader


class DataLoaderWrapper(object):
    """
    A minimal wrapper around PyTorch's DataLoader to return batches with the
    sequence dimension as the first dimension.

    Args:
        dataset (Dataset): the data to be loaded
    """
    def __init__(self, dataset, **kwargs):
        self._data_loader = DataLoader(dataset, kwargs)

    def __iter__(self):
        pass

    def __len__(self):
        return len(self._data_loader)
