import pickle
import numpy as np
import os

############################
### basketball constants ###
############################

LENGTH = 94
WIDTH = 50

###############################
### data-specific constants ###
###############################

TRAIN = 'train'
TEST = 'test'
BALL = 'ball'
OFFENSE = 'offense'
DEFENSE = 'defense'

N_TRAIN = 107146
N_TEST = 13845
SCALE = 10
SEQUENCE_LENGTH = 50
SEQUENCE_DIMENSION = 22

PLAYER_TYPES = [BALL, OFFENSE, DEFENSE]

COORDS = {
	BALL : { 'xy' : [0,1] },
	OFFENSE : { 'xy' : [2,3,4,5,6,7,8,9,10,11] },
	DEFENSE : { 'xy' : [12,13,14,15,16,17,18,19,20,21] }
}

for pt in PLAYER_TYPES:
	COORDS[pt]['x'] = COORDS[pt]['xy'][::2]
	COORDS[pt]['y'] = COORDS[pt]['xy'][1::2]

CMAP_ALL = ['orange'] + ['b']*5 + ['r']*5
CMAP_OFFENSE = ['b', 'r', 'g', 'm', 'y']

NORMALIZE = [LENGTH, WIDTH] * int(SEQUENCE_DIMENSION/2)
SHIFT = [25] * SEQUENCE_DIMENSION

###############################

def convert(path):
    """
    Converts the text files containing the basketball data into pickle files
    containing numpy arrays.

    Args:
        path (str): path to the basketball data directory
    """
    T = SEQUENCE_LENGTH
    for split in ['train', 'test']:

        os.makedirs(os.path.join(path, split))
        file_name = 'Xtr_role' if split == 'train' else 'Xte_role'

        N = N_TRAIN if split == 'train' else N_TEST
        data = np.zeros((N, T, SEQUENCE_DIMENSION))

        counter = 0
        f = open(os.path.join(path, file_name +'.txt'))
        for line in f:
            t = counter % T
            s = int((counter - t) / T)
            data[s][t] = line.strip().split(' ')
            counter += 1

        # get just the offensive players
        inds_data = COORDS['offense']['xy']
        data = data[:,:,inds_data]

        # convert multi-agent into single-agent data
        data = np.swapaxes(data, 0, 1)
        data = np.reshape(data, (len(data), -1, 2))
        data = np.swapaxes(data, 0, 1)

        # remove trajectories that were truncated incorrectly (about 3%)
        vel = data[:,1:,:]-data[:,:-1,:]
        speed = np.linalg.norm(vel, axis=2)
        max_speed = np.amax(speed, axis=1)
        good_inds = np.where(max_speed < 4)
        data = data[good_inds]

        pickle.dump(data, open(os.path.join(path, split, file_name + '.p'), 'wb'))
