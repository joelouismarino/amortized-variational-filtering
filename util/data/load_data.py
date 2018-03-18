from load_dataset import load_dataset
from torch.utils.data import DataLoader


def load_data(data_config, run_config, num_workers=4, pin_memory=True):
    """
    Wrapper around load_dataset. Gets the dataset, then places it in a DataLoader.
    """
    train, val, test = load_dataset(data_config, run_config)

    if train is not None:
        train = DataLoader(train, batch_size=run_config['batch_size'], shuffle=True,
                           num_workers=num_workers, pin_memory=pin_memory)

    if val is not None:
        val = DataLoader(val, batch_size=run_config['batch_size'], num_workers=num_workers,
                         pin_memory=pin_memory)

    if test is not None:
        test = DataLoader(test, batch_size=run_config['batch_size'], num_workers=num_workers,
                          pin_memory=pin_memory)

    return train, val, test
