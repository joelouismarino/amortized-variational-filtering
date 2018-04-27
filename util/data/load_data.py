from load_dataset import load_dataset
from transposed_collate import transposed_collate
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader


def load_data(data_config, batch_size, num_workers=4, pin_memory=True, sequence=True):
    """
    Wrapper around load_dataset. Gets the dataset, then places it in a DataLoader.

    Args:
        data_config (dict): data configuration dictionary
        batch_size (dict): run configuration dictionary
        num_workers (int): number of threads of multi-processed data Loading
        pin_memory (bool): whether or not to pin memory in cpu
        sequence (bool): whether data examples are sequences, in which case the
                         data loader returns transposed batches with the sequence
                         step as the first dimension and batch index as the
                         second dimension
    """
    train, val, test = load_dataset(data_config)

    if sequence:
        collate_func = transposed_collate
    else:
        collate_func = default_collate

    if train is not None:
        train = DataLoader(train, batch_size=batch_size, shuffle=True,
                           collate_fn=collate_func, num_workers=num_workers,
                           pin_memory=pin_memory)

    if val is not None:
        val = DataLoader(val, batch_size=batch_size,
                         collate_fn=collate_func, num_workers=num_workers,
                         pin_memory=pin_memory)

    if test is not None:
        test = DataLoader(test, batch_size=batch_size,
                          collate_fn=collate_func, num_workers=num_workers,
                          pin_memory=pin_memory)

    return train, val, test
