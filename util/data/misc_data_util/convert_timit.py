# Parts taken from https://github.com/marcofraccaro/srnn/blob/master/timit_for_srnn.py
# This script takes the path to the unzipped timit folder and converts the data
# into train, valid, and test splits.

import os
import fnmatch
import random
random.seed(1234)
import numpy as np
import cPickle
from config import run_config
try:
    import librosa
except ImportError:
    raise ImportError('Converting TIMIT dataset requires the librosa library. '+ \
                       'Please install librosa by running pip install librosa')

VAL_FRAC = 0.05
SAMPLING_RATE = 16000

def get_files(folder, file_end):
    # Recursively find files in folder structure with the file ending file_end
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, file_end):
            matches.append(os.path.join(root, filename))
    return matches

def load_wav_files(files, sampling_rate):
    wav_files = []
    for i, f in enumerate(files):
        print i, f
        wav_files += [librosa.load(f, sr=sampling_rate)[0]]
    return wav_files


def convert(data_path):

    m = std = None

    for data_split in ['train', 'test']:
        print('Converting ' + data_split)
        split_path = os.path.join(data_path, 'data', 'lisa', 'data', 'timit',
                                  'raw', 'TIMIT', data_split.upper())
        wav_files = get_files(split_path, '*.WAV')

        if data_split == 'train':
            # split the training set into train and validation
            num_train_files = len(wav_files)
            num_valid_split_files = int(num_train_files * VAL_FRAC)
            num_train_split_files = num_train_files - num_valid_split_files

            # shuffle the train files and split into train / val
            random.shuffle(wav_files)
            valid_split_files = wav_files[:num_valid_split_files]
            train_split_files = wav_files[num_valid_split_files:]

            # convert .WAV files into numpy arrays
            valid_vector = load_wav_files(valid_split_files, SAMPLING_RATE)
            train_vector = load_wav_files(train_split_files, SAMPLING_RATE)

            # concatenate all of the sequences together
            train_stack = np.hstack(train_vector)

            # train normalization statistics
            m = np.mean(train_stack)
            std = np.std(train_stack)

            # write the data to file
            os.makedirs(os.path.join(data_path, 'train'))
            cPickle.dump(train_vector, open(os.path.join(data_path, 'train', 'train.p'), 'w'))

            os.makedirs(os.path.join(data_path, 'val'))
            cPickle.dump(valid_vector, open(os.path.join(data_path, 'val', 'val.p'), 'w'))

        elif data_split == 'test':
            test_split_files = wav_files

            # convert .WAV files into numpy arrays
            test_vector = load_wav_files(test_split_files, SAMPLING_RATE)

            os.makedirs(os.path.join(data_path, 'test'))
            cPickle.dump(test_vector, open(os.path.join(data_path, 'test', 'test.p'), 'w'))

    cPickle.dump((m, std), open(os.path.join(data_path, 'statistics.p'), 'w'))
