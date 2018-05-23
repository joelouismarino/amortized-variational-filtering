import numpy as np


def get_max_min_notes(piano_roll_dict):
    """
    Gets the maximum number of notes across the entire data set.

    Args:
        piano_roll_dict (dict): dictionary containing piano rolls
    """
    max_note = 0
    min_note = 1e8
    for dataset in piano_roll_dict:
        unrolled = unroll(piano_roll_dict[dataset])
        max_note = max(np.max(unrolled), max_note)
        min_note = min(np.min(unrolled), min_note)
    return max_note, min_note


def unroll(piano_roll):
    """
    Unrolls a piano roll into a list of notes.

    Args:
        piano_roll (list of lists of lists): the piano roll list
    """
    temp = []
    for song in piano_roll:
        for notes in song:
            temp += notes
    return temp


def convert_midi(item, max_min_notes):
    """
    Converts midi music from piano roll format to numpy array.

    Args:
        item (list of lists): contains one song
        max_min_notes (tuple of ints): range of notes to allocate (max, min)
    """
    n_notes = max_min_notes[0] - max_min_notes[1] + 1
    array_item = np.zeros((len(item), n_notes)).astype('float32')
    for seq_ind, notes in enumerate(item):
        if len(notes) > 0:
            numpy_notes = np.array(notes) - max_min_notes[1]
            array_item[seq_ind, numpy_notes] = 1.
    return array_item


def filter_by_length(piano_roll, seq_len):
    new_piano_roll = []
    for song in piano_roll:
        if len(song) >= seq_len:
            new_piano_roll.append(song)
    return new_piano_roll
