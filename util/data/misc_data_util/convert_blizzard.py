import os
import fnmatch
import random
random.seed(1234)
import numpy as np
import cPickle
try:
    import librosa
except ImportError:
    raise ImportError('Converting Blizzard dataset requires the librosa library. '+ \
                       'Please install librosa by running pip install librosa')

VAL_FRAC = 0.05
TEST_FRAC = 0.05
SAMPLING_RATE = 16000 # Hz
CLIP_LEN = 1.0 # seconds
SAMPLE_LEN = int(CLIP_LEN * SAMPLING_RATE)

def get_files(folder, file_end):
    # Recursively find files in folder structure with the file ending file_end
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, file_end):
            matches.append(os.path.join(root, filename))
    return matches

def get_statistics(path):
    # NOTE: this calculates the mean of the means and stds (incorrect)
    npy_files = get_files(os.path.join(path, 'train'), '*.npy')

    m = np.zeros(len(npy_files))
    s = np.zeros(len(npy_files))

    for i, f in enumerate(npy_files):
        clip = np.load(f)
        m[i] = np.mean(clip)
        s[i] = np.std(clip)

    m = m.mean()
    s = s.mean()

    cPickle.dump((m, s), open(os.path.join(path, 'statistics.p'), 'w'))

def convert(data_path):

    # get all of the mp3 files
    mp3_files = get_files(os.path.join(data_path, 'data', 'train'), '*.mp3')

    os.makedirs(os.path.join(data_path, 'train'))
    os.makedirs(os.path.join(data_path, 'val'))
    os.makedirs(os.path.join(data_path, 'test'))

    for i, f in enumerate(mp3_files):
        print(i, f)
        # load the file
        f_data = librosa.load(f, sr=SAMPLING_RATE)[0]

        # split into clips
        for ind, start in enumerate(range(0, f_data.shape[0], SAMPLE_LEN)[:-1]):
            clip = f_data[start:start+(SAMPLE_LEN)]

            # assign each clip to train/val/test
            split = None
            r = random.random()
            if r >= 1 - TEST_FRAC:
                split = 'test'
            elif r >= 1 - TEST_FRAC - VAL_FRAC:
                split = 'val'
            else:
                split = 'train'

            # write to corresponding directory
            clip_name = os.path.join(data_path, split, 'file_' + str(i) + '_clip_' + str(ind) + '.npy')
            np.save(clip_name, clip)

    get_statistics(data_path)
