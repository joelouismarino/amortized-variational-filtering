import numpy as np
from util.data.misc_data_util.midi_util import convert_midi
from torch.utils.data import Dataset


class MIDI(Dataset):
    """
    Dataset for MIDI music data.

    Args:
        piano_roll (list of lists): midi data in piano_roll format
        transform (transforms): function that applies any data transforms
    """
    def __init__(self, piano_roll, max_min_notes, transform=None):
        self.piano_roll = piano_roll
        self.max_min_notes = max_min_notes
        self.transform = transform

    def __getitem__(self, ind):
        item = self.piano_roll[ind]
        item = convert_midi(item, self.max_min_notes)
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __len__(self):
        return len(self.piano_roll)
