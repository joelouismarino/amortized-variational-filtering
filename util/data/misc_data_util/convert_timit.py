# Largely taken from https://github.com/marcofraccaro/srnn/blob/master/timit_for_srnn.py
# This script takes the path to the unzipped timit folder and converts the data
# into train, valid, and test splits. The input is processed as frames that
# consist of 200 consecutive raw audio inputs. These are then strung together
# for SEQ_LEN steps.

# TODO: save each WAV file individually so that we can sample different windows
#       during training and not get rid of examples

import os
import fnmatch
import random
random.seed(1234)
import numpy as np
from config import run_config
try:
    import librosa
except ImportError:
    raise ImportError('Converting TIMIT dataset requires the librosa library. '+ \
                       'Please install librosa by running pip install librosa')

BATCH_SIZE = run_config['batch_size']
VAL_FRAC = 0.05
SAMPLING_RATE = 16000
SEQ_LEN_IN_SEC = 0.5
OUT_DIM = 200
SEQ_LEN = int(SAMPLING_RATE * SEQ_LEN_IN_SEC) / OUT_DIM

def get_files(folder, file_end):
    # Recursively find files in folder structure with the file ending file_end
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, file_end):
            matches.append(os.path.join(root, filename))
    return matches

def load_wav_files(files):
    wav_files = []
    for i, f in enumerate(files):
        print i, f
        wav_files += [librosa.load(f, sr=SAMPLING_RATE)[0]]
    return wav_files

def make_multiple(x, outdim):
    n = int(len(x) // outdim)
    out_len = n*outdim
    return x[:out_len]

def reorder(data_in, batch_size, model_seq_len, dtype='float32'):
    last_dim = data_in.shape[-1]
    if data_in.shape[0] % (batch_size * model_seq_len) == 0:
        print(" x_in.shape[0] % (batch_size*model_seq_len) == 0 -> x_in is "
              "set to x_in = x_in[:-1]")
        data_in = data_in[:-1]

    data_resize = \
        (data_in.shape[0] // (batch_size * model_seq_len)) * model_seq_len * batch_size
    n_samples = data_resize // (model_seq_len)
    n_batches = n_samples // batch_size

    u_out = data_in[:data_resize].reshape(n_samples, model_seq_len, last_dim)
    x_out = data_in[1:data_resize + 1].reshape(n_samples, model_seq_len, last_dim)

    out = np.zeros(n_samples, dtype='int32')
    for i in range(n_batches):
        val = range(i, n_batches * batch_size + i, n_batches)
        out[i * batch_size:(i + 1) * batch_size] = val

    u_out = u_out[out]
    x_out = x_out[out]

    return u_out.astype(dtype), x_out.astype(dtype)

def create_test_set(x_lst):
    n = len(x_lst)
    x_lens = np.array(map(len, x_lst))
    max_len = max(map(len, x_lst)) - 1
    u_out = np.zeros((n, max_len, OUT_DIM), dtype='float32')*np.nan
    x_out = np.zeros((n, max_len, OUT_DIM), dtype='float32')*np.nan
    for row, vec in enumerate(x_lst):
        l = len(vec) - 1
        u = vec[:-1]  # all but last element
        x = vec[1:]   # all but first element

        x_out[row, :l] = x
        u_out[row, :l] = u

    mask = np.invert(np.isnan(x_out))
    x_out[np.isnan(x_out)] = 0
    u_out[np.isnan(u_out)] = 0
    mask = mask[:, :, 0]
    assert np.all((mask.sum(axis=1)+1) == x_lens)
    return u_out, x_out, mask.astype('float32')


def convert(data_path):

    m = sd = None

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
            valid_vector = load_wav_files(valid_split_files)
            train_vector = load_wav_files(train_split_files)

            # concatenate all of the sequences together
            train_vector = np.hstack(train_vector)
            valid_vector = np.hstack(valid_vector)

            # normalization according to train statistics
            m = np.mean(train_vector)
            sd = np.std(train_vector)

            train_vector = (train_vector - m) / sd
            valid_vector = (valid_vector - m) / sd

            # make train and valid vectors a multiple of the output dimension
            train_vector = make_multiple(train_vector, OUT_DIM).reshape((-1, OUT_DIM))
            valid_vector = make_multiple(valid_vector, OUT_DIM).reshape((-1, OUT_DIM))

            # convert into sequences
            _, x_train_vector = reorder(train_vector, BATCH_SIZE, SEQ_LEN)
            _, x_valid_vector = reorder(valid_vector, BATCH_SIZE, SEQ_LEN)

            # write the arrays to file
            os.makedirs(os.path.join(data_path, 'train'))
            np.savez_compressed(os.path.join(data_path, 'train', 'train.npz'),
                                x=x_train_vector)
            os.makedirs(os.path.join(data_path, 'val'))
            np.savez_compressed(os.path.join(data_path, 'val', 'val.npz'),
                                x=x_valid_vector)

        elif data_split == 'test':
            test_split_files = wav_files

            # convert .WAV files into numpy arrays
            test_vector_lst = load_wav_files(test_split_files)

            # normalization according to train statistics
            test_vector_lst = [(vec - m)/sd for vec in test_vector_lst]

            # make each vector a multiple of the output dimension
            test_vector_lst = [make_multiple(v, OUT_DIM).reshape((-1, OUT_DIM)) for v in test_vector_lst]

            u_test_vector, x_test_vector, mask_test = create_test_set(test_vector_lst)
            os.makedirs(os.path.join(data_path, 'test'))
            np.savez_compressed(os.path.join(data_path, 'test', 'test.npz'),
                                x=x_test_vector,
                                mask=mask_test)
